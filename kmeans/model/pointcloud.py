from dataclay import DataClayObject, dclayMethod
from dataclay.contrib.splitting import SplittableCollectionMixin

class PointCloud(DataClayObject, SplittableCollectionMixin):
    """
    @ClassField chunks list<storageobject>
    """
    @dclayMethod()
    def __init__(self):
        self.chunks = list()
    
    @dclayMethod(fragment="storageobject")
    def add_fragment(self, fragment):
        self.chunks.append(fragment)
