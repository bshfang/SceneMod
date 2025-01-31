
from datasets.threed_front_graph_mod_s1 import Threed_Front_Graph_Mod_Stage_One
from datasets.threed_front_graph_mod_s2 import Threed_Front_Graph_Mod_Stage_TWO

def build_dataset(cfg, split=['train'], inference=False):
    dataset_name = cfg['data']['dataset_name']
    
    if dataset_name == "3dfront_graph_mod_s1": # s1
        dataset = Threed_Front_Graph_Mod_Stage_One(cfg['data'],
                                    splits=split,
                                    inference=inference)
    elif dataset_name == "3dfront_graph_mod_s2": # s2
        dataset = Threed_Front_Graph_Mod_Stage_TWO(cfg['data'],
                                    splits=split,
                                    inference=inference)

    
    return dataset