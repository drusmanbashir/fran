import ast
import itertools as il

ast_keys = ["spacing", "spatial_size"]
class DictToAttr(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def assimilate_dict(self,dic:dict):
        for key,val in dic.items():
            setattr(self,key,val)
 
    @classmethod
    def from_nested_dicts(cls, data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return cls({key: cls.from_nested_dicts(data[key]) for key in data})


def key_from_value(dici:dict, value):
    keys = []
    for k,v in dici.items():
        if v == value:
            keys.append(k)
    return keys

def assimilate_dict(cls,dic:dict):
        objectified_dic = DictToAttr.from_nested_dicts(dic)
        for key,val in objectified_dic.items():
            setattr(cls,key,val)
        return cls

def list_unique(values:list):
     return list(set(values))


def dic_lists_to_sets(dic:dict)->dict:
        switcher= lambda key,val: [key,set(val)] if isinstance(val,list) else [key,val]
        dic_out={}
        for key,val in dic.items():
            outs = switcher(key,val)
            dic_out.update({outs[0]:outs[1]})
        return dic_out

def dic_in_list(query_dic,dic_list)->bool:
    query_dic = dic_lists_to_sets(query_dic)
    dic_list= [dic_lists_to_sets(dic) for dic in dic_list]
    return True if query_dic in dic_list else False


def fix_ast(dici ):
    keys = dici.keys()
    # Use itertools and functools to filter keys
    relevant_keys = list(il.compress(keys, (key in ast_keys and key in dici for key in keys)))
    for key in relevant_keys:
        if isinstance(dici[key], str):
            dici[key] = ast.literal_eval(dici[key])
    
    return dici


def fix_ast_nested_dicts(data):
    if isinstance(data, dict):
        # Apply fix_ast to the current dictionary.
        fix_ast(data)
        # Recursively apply fix_ast_nested_dicts to nested dictionaries.
        for key, value in data.items():
            if isinstance(value, dict):
                fix_ast_nested_dicts(value)
    return data


