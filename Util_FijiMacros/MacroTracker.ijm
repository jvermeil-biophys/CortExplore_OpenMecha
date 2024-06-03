//dir = "D:/MagneticPincherData/Raw/24.05.24_Chameleon/M2-20um/Bstairs/Clean_Fluo_BF";
//dir = "D:/MagneticPincherData/Raw/24.05.24_Chameleon_depthos/M2";
dir="C:/Users/JosephVermeil/Desktop/CortExplore_OpenMecha/Example_Data_2024/05_ExampleCell3";
run("Set Measurements...", "area standard center fit stack redirect=None decimal=3");
name = getInfo("image.title");
resultsName = substring(name, 0, name.length - 4) + "_Results.txt";
//Shrimp Parms
run("Analyze Particles...", "size=100-2000 circularity=0.60-1.00 show=Outlines display exclude clear include stack");
//Chameleon Parms
//run("Analyze Particles...", "size=40-500 circularity=0.40-1.00 show=Outlines display exclude clear include stack");
saveAs("Results", dir + "/" + resultsName);