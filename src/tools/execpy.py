def execpy(path):
    exec(open(path).read(),globals())
