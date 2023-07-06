import random
list=[]
w=open("jingbo_test.list","a")
with open("FlyingThings3D_release_TEST.list","r",encoding="utf-8") as f:
    lines=f.readlines()
    for i in random.sample(range(0,4370),1000):
        w.write(lines[i])
    w.close()

    f.close()

