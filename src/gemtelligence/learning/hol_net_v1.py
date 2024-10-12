from .base_model import BaseModel
from .saint import SAINT
from .resnet import ResNet

from torch import nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import hydra
from pathlib import Path
import pickle


def load_model(cfg, val_path=None):
    if val_path:
        if "ed" in cfg.sources or "icp" in cfg.sources:
            first_batch = return_first_batch_from_model_path(val_path)
        else:
            first_batch = None

        # Move it to the GPU
        # first_batch = {k: v.cuda() for k, v in first_batch.items()}
        model = HolNetV1.load_from_checkpoint(
            val_path, cfg=cfg, first_batch=first_batch
        )
        # Delete the first batch to reduce the memory usage
        del first_batch

    else:
        model = HolNetV1(cfg)
    return model


def return_serialized_model(model_path):
    with open(model_path, "rb") as f:
        serial_model = f.read()
    return serial_model


def return_first_batch_from_model_path(model_path, is_raw=False):
    first_batch_path_test_one = Path(
        hydra.utils.to_absolute_path(Path(model_path).parent / "0" / "first_batch.pkl")
    )
    second_batch_path_test_second = Path(
        hydra.utils.to_absolute_path(
            Path(model_path).parent.parent.parent.parent / "first_batch.pkl"
        )
    )
    if first_batch_path_test_one.exists():
        first_batch_path = first_batch_path_test_one
    elif second_batch_path_test_second.exists():
        first_batch_path = second_batch_path_test_second
    if is_raw:
        with open(first_batch_path, "rb") as f:
            serial_first_batch = f.read()
        return serial_first_batch

    with open(first_batch_path, "rb") as f:
        first_batch = pickle.load(f)
    return first_batch


class HolNetV1(BaseModel):
    """
    Combine ED and UV data together
    """

    def get_encoding(self, input_lenght):
        out = self.ftir_reductor(torch.ones(1, 1, input_lenght))
        return out.shape[2]

    def __init__(self, cfg=None, first_batch=None):
        super().__init__(cfg=cfg, first_batch=first_batch)
        self.number_of_classes = cfg.class_len
        self.source_encoder = {}
        self.source_reductor = {}
        self.end_dims = {}

        if "uv" in cfg.sources.keys():
            input_channels = cfg.sources.uv.data_processing + 1
            if not cfg.method.all_saint:
                self.uv_model = ResNet(
                    cfg=cfg.method.model.resnet,
                    input_length=cfg.sources.uv.input_dim,
                    input_channels=input_channels,
                )
            self.uv_reductor = nn.Conv1d(
                self.uv_model.z_inner_dim.shape[1], 5, kernel_size=1
            )
            self.source_encoder["uv"] = self.uv_model
            self.source_reductor["uv"] = self.uv_reductor
            self.end_dims["uv"] = 5 * self.uv_model.z_inner_dim.shape[2]

        if "ftir" in cfg.sources.keys():
            input_channels = cfg.sources.ftir.data_processing + 1
            self.ftir_model = ResNet(
                cfg=cfg.method.model.resnet,
                input_length=cfg.sources.ftir.input_dim,
                input_channels=input_channels,
            )
            stride = 1
            channels = 1
            self.ftir_reductor = nn.Conv1d(
                self.ftir_model.z_inner_dim.shape[1],
                channels,
                kernel_size=1,
                stride=stride,
            )
            end_dim = channels * (self.ftir_model.z_inner_dim.shape[2] // stride)
            self.source_encoder["ftir"] = self.ftir_model
            self.source_reductor["ftir"] = self.ftir_reductor
            self.end_dims["ftir"] = end_dim

        if "ed" in cfg.sources.keys() or "icp" in cfg.sources.keys():

            # Make sure not both tab_transformer and saint are used at the same time
            assert not (
                ("tab_transformer" in cfg.method.model) & ("saint" in cfg.method.model)
            )

            if "saint" in cfg.method.model:
                if "icp" in cfg.sources.keys() and "ed" in cfg.sources.keys():
                    num_columns = (
                        cfg.sources.icp.num_columns + cfg.sources.ed.num_columns
                    )
                elif "icp" in cfg.sources.keys() and (not "ed" in cfg.sources.keys()):
                    num_columns = cfg.sources.icp.num_columns
                else:
                    num_columns = cfg.sources.ed.num_columns

                # Dummy cfg copy to pass the configuration
                cfg_copy = cfg.copy()
                cfg_copy.method = cfg.method.model.saint
                self.ed_model = SAINT(
                    categories=tuple(np.array([1]).astype(int)),
                    num_continuous=num_columns,
                    dim=cfg.method.model.saint.embedding_size,
                    dim_out=1,
                    depth=cfg.method.model.saint.transformer_depth,
                    heads=cfg.method.model.saint.attention_heads,
                    attn_dropout=cfg.method.model.saint.attention_dropout,
                    ff_dropout=cfg.method.model.saint.ff_dropout,
                    mlp_hidden_mults=(4, 2),
                    cont_embeddings=cfg.method.model.saint.cont_embeddings,
                    attentiontype=cfg.method.model.saint.attentiontype,
                    final_mlp_style=cfg.method.model.saint.final_mlp_style,
                    y_dim=cfg.class_len,
                    cfg=cfg,
                )
                self.source_encoder["ed"] = self.ed_model
                self.source_reductor["ed"] = nn.Identity()
                self.end_dims["ed"] = cfg_copy.method.embedding_size

        self.batch_norm = nn.BatchNorm1d(sum(self.end_dims.values()))
        self.linear_classifier = nn.Linear(
            sum(self.end_dims.values()),
            self.number_of_classes + cfg.secondary_class_len,
        )
        self.lr = cfg.method.lr

    def model(self, x):
        return self.forward(x, is_already_padded=True)

    def forward(self, x, is_already_padded=False):
        if type(x) == np.ndarray:
            raise Exception("Numpy arrays are not supported")

        if type(x) != torch.Tensor and self.first_batch:
            for key in x.keys():
                if key in ["ed", "icp"]:
                    if self.first_batch[key].shape[0] > 0:
                        place_holder = self.first_batch[key].mean(axis=0)
                        if x[key].is_cuda and not place_holder.is_cuda:
                            place_holder = place_holder.cuda()
                        bool_cond = x[key].sum(axis=1) == 0
                        x[key][bool_cond] = place_holder
        # Ugly hack only for capsule
        elif type(x) == torch.Tensor:
            x_method = x.clone()
            self.cfg.method.model.saint.fake_batch = x_method.shape[0]

            x = {}
            if "uv" in self.cfg.sources:
                if x_method.shape[2] == 1201:
                    x_uv = x_method
                else:
                    x_uv = torch.zeros(x_method.shape[0], 2, 1201, device=self.device)
                x_uv = self.source_encoder["uv"].encode(x_uv)
                x_uv = self.source_reductor["uv"](x_uv).flatten(start_dim=1)
                x["uv"] = x_uv

            if "ftir" in self.cfg.sources:
                if x_method.shape[2] == self.cfg.sources.ftir.input_dim:
                    x_ftir = x_method
                else:
                    x_ftir = torch.zeros(
                        x_method.shape[0],
                        1,
                        self.cfg.sources.ftir.input_dim,
                        device=self.device,
                    )

                x_ftir = self.source_encoder["ftir"].encode(x_ftir)
                x_ftir = self.source_reductor["ftir"](x_ftir).flatten(start_dim=1)
                x["ftir"] = x_ftir

            tmp = [x[key] for key in sorted(x.keys()) if key != "icp"]
            embedding = torch.cat((tmp), dim=1)
            embedding = self.batch_norm(embedding)

            return self.linear_classifier(embedding)

        true_batch_size = min(x.shape[0] for x in x.values())
        keys = x.keys()
        for source in keys:
            if source in ["uv", "ftir"]:
                x[source] = self.source_encoder[source].encode(x[source])
                x[source] = self.source_reductor[source](x[source]).flatten(start_dim=1)
            elif source == "ed":  # if icp is used, it is passed together with ed

                if "saint" in self.cfg.method.model:
                    if is_already_padded == True:
                        _, x[source] = self.source_encoder[source](
                            x, padding=None, device=self.device
                        )
                    else:
                        _, x[source] = self.source_encoder[source](
                            x, padding=self.first_batch, device=self.device
                        )
                    # x[source] is 26 element, make it to be 32
                x[source] = self.source_reductor[source](x[source])[:true_batch_size]
            elif source == "icp" and not "ed" in keys:
                if "saint" in self.cfg.method.model:
                    if is_already_padded == True:
                        _, x[source] = self.source_encoder["ed"](
                            x, padding=None, device=self.device
                        )
                    else:
                        _, x[source] = self.source_encoder["ed"](
                            x, padding=self.first_batch, device=self.device
                        )
                    # x[source] is 26 element, make it to be 32
                x[source] = self.source_reductor["ed"](x[source])[:true_batch_size]
                # pass # included in Saint
            elif source == "icp" and "ed" in keys:
                pass  # Included in line 175
            else:
                raise KeyError

        if ("icp" in self.cfg.sources.keys()) and (not "ed" in self.cfg.sources.keys()):
            x["ed"] = x["icp"]
            del x["icp"]

        tmp = [x[key] for key in sorted(x.keys()) if key != "icp"]
        embedding = torch.cat((tmp), dim=1)
        embedding = self.batch_norm(embedding)

        return self.linear_classifier(embedding)
