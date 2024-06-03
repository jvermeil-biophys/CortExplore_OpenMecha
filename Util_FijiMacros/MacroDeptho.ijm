// DO NOT FORGET TO SET THE SCALE HERE !
// AND TO REMOVE THE SCALE FROM THE IMAGE (we work with pixel units)
SCALE = 15.8; // Pix/um
//SCALE = 7.4588; // Pix/um

rect_h = round(6.3 * SCALE);
rect_h = rect_h + rect_h%2;
rect_w = round(4.7 * SCALE);
rect_w = rect_w + rect_h%2;

dir = getDirectory("Select a Directory")
list = getFileList(dir);
run("Set Measurements...", "area standard center stack redirect=None decimal=3");

for (i = 0; i < list.length; i++) {
	open(dir + "/" + list[i]);
	selectImage(list[i]);
	run("In [+]");
	run("In [+]");
	ny = getHeight();
	nx = getWidth();
	makeRectangle(nx/2 - rect_w/2, ny/2 - rect_h/2, rect_w, rect_h);
	run("Threshold...");
	setAutoThreshold("Default dark no-reset stack");
	selectImage(list[i]);
	run("Analyze Particles...", "size=40-2000 pixel circularity=0.20-1.00 display exclude clear include stack");
	saveAs("Results", dir + "/" + substring(list[i], 0, list[i].length - 4) + "_Results.txt");
}

run("Close All");