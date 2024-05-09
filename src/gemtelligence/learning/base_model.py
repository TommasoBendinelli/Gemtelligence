from torch import nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import hydra
import sklearn
from collections import defaultdict
import os

class BaseModel(pl.LightningModule):
    """
    Model that is inherited from all neural network models.
    It is used to define the model and the training loop.
    """
    def __init__(
        self,
        cfg=None,
        first_batch=None
    ):
        super().__init__()

        if cfg.target == "treatment" and cfg.method.balanced_loss:
            weightes = torch.tensor([1.0,20.0])
        else:
            weightes = None
        self.criterion = nn.CrossEntropyLoss(weight=weightes)
        
        if cfg.method.val_method == "max":
            self.best_shot = {}

        elif cfg.method.val_method == "mean":
            self.confidence_list = defaultdict(list)
        else:
            raise NotImplementedError
        
        self.confidence = {}
        self.gt = {}
        self.cfg = cfg
        self.is_trained = False
        if first_batch:
            self.first_batch = {key: val[:cfg.method.model.saint.fake_batch,:] for key, val in first_batch.items()} 
            self.is_filled = True
        else:
            self.first_batch = None
            self.is_filled = False
        self.is_saved = False
        #

    def training_step(self, batch, batch_idx):
        
        if "fake_batch" in self.cfg.method:
            fb = self.cfg.method.fake_batch
        else:
            fb = self.cfg.method.model.saint.fake_batch
            
        sources_first_batch = sorted(list({"ed","icp"} & set(batch[0].keys())))
        # training_step defined the train loop. It is independent of forward
        self.is_trained = True
        x, y, y_sec, ID = batch
        if "ed" in x.keys():
            key_for_fake = "ed"

        #go in this block only in the first batch of training
        if not self.first_batch:
            self.first_batch = {}
            for key in sources_first_batch:
                val = x[key]
                # Sum all but the batch dimension
                bool_cond =  val.sum(dim=[x for x in  range(1,len(val.shape))]) != 0
                not_zeros_val = val[bool_cond]
                self.first_batch[key] = not_zeros_val

        #should be true after first batch
        elif not self.is_filled:
            # Combine first batch with current batch
            for key in sources_first_batch:
                val = x[key]
                bool_cond = val.sum(dim=[x for x in range(1,len(val.shape))]) != 0
                not_zeros_val = val[bool_cond]          
                self.first_batch[key] = (torch.cat((self.first_batch[key], not_zeros_val), dim=0)[:fb])
                
            # Check if all sources are filled
            try_is_filled = True
            for key in sources_first_batch:
                try_is_filled = (self.first_batch[key].shape[0] >= fb) and try_is_filled

            self.is_filled = try_is_filled
                    
                
        if (self.is_filled) and (not self.is_saved):
            # Create a cpu copy of the first batch
            copy_first_batch = {key: val.cpu() for key, val in self.first_batch.items()}
            # Save the copy_first_batch
            with open(os.path.join("first_batch.pkl"), "wb") as f:
                import pickle
                pickle.dump(copy_first_batch, f)
            self.is_saved = True
        else:
            pass
        
        min_dim = float("+inf")
        for key in x.keys():
            if key in sources_first_batch:
                # Permute self.first_batch randomly
                self.first_batch[key] = self.first_batch[key]#[torch.randperm(self.first_batch[key].shape[0])]
                x[key] = torch.cat((x[key], self.first_batch[key]), dim=0)[:fb]
                min_dim = min(x[key].shape[0],min_dim)

        for key in x.keys():
              if key in sources_first_batch:
                  x[key] = x[key][:min_dim]
    
        for key in x.keys():
            if key in sources_first_batch:
                assert x[key].shape[0] == min_dim

        # Replacing all zeros with average from pre-cached elements      
        # for key in x.keys():
        #     if key in sources_first_batch:
        #         if self.first_batch[key].shape[0] > 0:
        #             place_holder = self.first_batch[key].mean(axis=0)
        #             bool_cond = x[key].sum(axis=1) == 0
        #             x[key][bool_cond] = place_holder
        if any([torch.isnan(value).any() for _, value in x.items()]):
            breakpoint()

        logits = self.model(x)[:y.shape[0]]
        # Check if any value is nan
        if torch.isnan(logits).any():
            breakpoint()
        
        loss = self.criterion(logits[:,:self.cfg.class_len], y.type(torch.long))
        
        bool_sec_mask = y_sec!=-1
        y_sec = y_sec[bool_sec_mask]
        logits2 = logits[bool_sec_mask,self.cfg.class_len:]
        
        # if self.cfg.add_secondary_target:
        #     loss = loss + self.criterion(logits2, y_sec.type(torch.long))
        
        
        # accu = sum(correct) / y.shape[0]
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "lr",
            self.lr,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        return loss
        # return None

    def validation_step(self, batch, batch_idx):
        # validation_step defined the train loop. It is independent of forward
        x, ys, y_secs, IDs = batch
        valbatch = {}
        if self.is_trained == False:
            for source in x:
                x[source] = torch.ones(size=x[source].shape, device=self.device)# Avoid passing 0 to model, it will crash with CUDA ERRORS
        
        first_key = list(x.keys())[0]
        valbatch = x[first_key].shape[0]
        if self.first_batch is not None:
            for source in x:
                if source in ["ed","icp"]:
                    to_pad = self.first_batch[source][:-valbatch]
                    x[source] = torch.cat((x[source],to_pad),dim=0)
        
        sources_first_batch = sorted(list({"ed","icp"} & set(batch[0].keys())))
                    
        # x = x.permute(0, 3, 1, 2)
        logits = self.model(x.copy())[:valbatch]
        loss = self.criterion(logits[:,:self.cfg.class_len], ys.type(torch.long))
        bool_sec_mask = y_secs!=-1
        y_secs = y_secs[bool_sec_mask]
        logits2 = logits[bool_sec_mask,self.cfg.class_len:]
        
        # if self.cfg.add_secondary_target:
        #     loss = loss + self.criterion(logits2, y_secs.type(torch.long))
        
        prob = torch.nn.functional.softmax(logits[:,:self.cfg.class_len],dim=1)

        for idx, ID in enumerate(IDs):
            if self.cfg.method.val_method == "max":
                v = float(torch.max(prob[idx]))
                if not int(ID) in self.best_shot or self.best_shot[int(ID)] < v:
                    self.best_shot[int(ID)] = v                
                    self.confidence[int(ID)] = np.array(prob[idx].cpu())
                    self.gt[int(ID)] = int(ys[idx])
            elif self.cfg.method.val_method == "mean":              
                self.confidence_list[int(ID)].append(np.array(prob[idx].cpu()))
                self.gt[int(ID)] = int(ys[idx])

        return loss

    def validation_epoch_end(self, res):
        IDs = list(self.gt.keys())
        gt = np.array([self.gt[ID] for ID in IDs])
        if self.cfg.method.val_method == "mean":
            self.confidence = {ID: np.mean(np.array(self.confidence_list[ID]),axis=0) for ID in IDs}

        probs = np.array([self.confidence[ID] for ID in IDs])
        print("Number of NaN: ", np.sum(np.isnan(probs).any(axis=1)))
        # Replace nan with 0
        probs[np.isnan(probs)] = 0

        pred = np.array([np.argmax(self.confidence[ID]) for ID in IDs])
        acc = sklearn.metrics.accuracy_score(gt,pred)
        loss = sklearn.metrics.log_loss(gt,probs,labels=list(range(self.number_of_classes)))
        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.best_shot = {}
        self.confidence = {}
        self.gt = {}
        if self.cfg.method.val_method == "mean":
            self.confidence_list = defaultdict(list)

    def configure_optimizers(self):
        if self.cfg.method.lr_type == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.5, 0.999))
        elif self.cfg.method.lr_type == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.cfg.method.lr_type == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        elif self.cfg.method.lr_type == "all":
            resnet_parameters = self.ftir_model.parameters()
            other_parameters = [x for x in list(self.parameters()) if x not in resnet_parameters]
            optimizer = torch.optim.AdamW([{"params": other_parameters, "lr":self.lr, "betas":(0.9, 0.999)},{"params": resnet_parameters, "lr":self.lr*10, "betas":(0.9, 0.999)}])
        else:
            raise KeyError("Unknown optimizer type")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=10, 
                                                        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
            
        return {"optimizer": optimizer, "lr_scheduler":scheduler, "monitor":"train_loss"} 