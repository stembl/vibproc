def delmodule(module_name):
    import sys
    if module_name in sys.modules:
        del sys.modules[module_name]
    del module_name

def reload(module_name):
    delmodule(module_name)

    __import__(module_name)
