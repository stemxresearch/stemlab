
def num_floatify():
    try:
        x = float(x)
    except:
        pass
    
    return x

def num_integify(x):
    
    try:
        x = int(x) if x % 1 == 0 else x
    except:
        pass
    
    return x