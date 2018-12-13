import pickle
from keras.models import load_model
from Serialized_Callback  import History_resumable as History
        
''' #Resumable Trainer with callbacks#
!!! The callbacks used should be 'picklable' !!!

-> A simple wrapper of current Keras.Model
-> Try to solve the problem that Keras does not store the
    states of callbacks through model.save, which really annoys
    people who have to resume training model frequently.
    
-> The trainer will train model from '@start_epoch'+1 to 
    final '@epochs', while stop every '@ep_turn'. People
    can use 'Pickle' outside to dump and load this trainer
    with trainning state automatically saved.
-> To start one turn of training, simply called trainer.train
    with user-maintained model, place to store the model, and
    the parameter for Model.fit except 'initial_epoch', 'epochs'
    , and 'callbacks'.  A simplified history instance is return
    similar to original Model.fit.
    [?] => The reason I leave model out for user to maintain is 
        to increase the flexibaility for user to change their
        models. A 'set_callbacks' function is provided for a
        similar reason.
-> A 'isStopped'function is offered to check stop.
'''        
class ResumableTrainer_callback:
    def __init__(self,epochs,ep_turn,pickle_path,start_epoch = 0,earlystop=False,callbacks=None,
                 custom_objects = None):
        self.epochs = epochs
        self.ep_turn = ep_turn
        self.pickle_path = pickle_path
        self.start_epoch = start_epoch
        self.history = History()
        self.callbacks = callbacks
        self.earlystop = earlystop
        self.custom_objects = custom_objects
        self.stopped = False
        
    def isStopped(self):
        return self.stopped
        
    def set_callbacks(self,callbacks,extend=True):
        if extend: self.callbacks.extend(callbacks)
        else: self.callbacks = callbacks
        
    def new_turn(self):
        self.start = self.start_epoch
        self.end = min(self.epochs, self.start+self.ep_turn)
    
    def train(self,model,filename,*para,**kpara):
        self.new_turn()
        model.fit(*para,**kpara,initial_epoch=self.start,epochs=self.end,callbacks = [self.history]+self.callbacks)
        self.stopped = self.end==self.epochs or self.earlystop and model.stop_training
        model.save(filename, overwrite=True)
        self.clear_callbacks()
        self.save_trainer()
        return self.history

    def train_generator(self, model, filename, *para, **kpara):
        self.new_turn()
        model.fit_generator(*para, **kpara, initial_epoch=self.start, epochs=self.end, callbacks=[self.history] + self.callbacks)

        self.stopped = self.end == self.epochs or self.earlystop and model.stop_training
        model.save(filename, overwrite=True)
        self.clear_callbacks()
        self.save_trainer()
        return self.history
    
    def clear_callbacks(self):
        callbacks = self.callbacks+[self.history]
        for i in callbacks:
            if i.validation_data: i.validation_data = None
            if i.model: i.model = None

    def save_trainer(self):
        if self.history.epoch: self.start_epoch = self.history.epoch[-1]+1
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(self, f)

    def load(self,path):
        return load_model(path, custom_objects=self.custom_objects)