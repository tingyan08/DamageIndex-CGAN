import os
import glob
import pandas as pd


label = {"filename":[], "DI":[], "AR":[], "VR":[], "HR":[]}

for i in os.listdir("./Plane_data_750x800/CTR1"):
    exp = i.split("_")[0]
    DI = float(i[-9:-6])
    if exp == "C307":
        AR = 3
        VR = 0.742
        HR = 1.282
    elif exp == "C315":
        AR = 3
        VR = 1.444
        HR = 1.106
    elif exp == "C330":
        AR = 3
        VR = 2.889
        HR = 1.106
    elif exp == "C615":
        AR = 6
        VR = 1.444
        HR = 1.106
    elif exp == "C1015":
        AR = 10
        VR = 1.444
        HR = 1.106

    elif exp == "CTR1":
        AR = 3.5
        VR = 2.140
        HR = 1.800

    label["filename"].append(i)
    label["DI"].append(DI)
    label["AR"].append(AR)
    label["VR"].append(VR)
    label["HR"].append(HR)        

df = pd.DataFrame(data=label, columns=["filename", "DI", "AR", "VR", "HR"])
df.to_csv("./Plane_data_750x800/CTR1.csv", index=False)
        

