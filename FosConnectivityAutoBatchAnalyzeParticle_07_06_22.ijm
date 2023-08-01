/*
 * 	Yihan Xie - Blendy Lab
 *	This Macro Script enables batch processing and analyze the particles with selected thresholding method.
 *	The results for each image data will be auto saved to given directory
 *	The ROIs will also be saved in the same folder
 *	directory is the folder path where all image saved
 *	To switch to other RGB channel imaged (by default green), add and delete close() below the selectWindow for different color channels
 */

directory = File.getParent(File.openDialog("Select one file in the Input batch Folder: "));
Dialog.create("Specify ROI name for this analysis");
Dialog.addMessage("Please provide the name of ROI, the result of ROI region data will be saved in folder with provided name.");
Dialog.addString("ROI Name: ", "DR");
Dialog.show();
ROI_Name = Dialog.getString();
File.makeDirectory(directory + File.separator + ROI_Name+" Analyzed Result");
File.makeDirectory(directory + File.separator + ROI_Name+" ROI Selected");
DestinationDirectory = directory + File.separator + ROI_Name+" Analyzed Result";
ROI_Directory_To_Save = directory + File.separator + ROI_Name+" ROI Selected";

Dialog.create("Specify Image Group and Subgroup Name");
Dialog.addMessage("Please specify your image naming, e.g., Brain 1 Slice 1 ~ Brain 3 Slice 3 will put Brain 1,Brain 2,Brain 3 in Sample Name; Slice 1,Slice 2,Slice 3 in Group Name. Use , with no space in between.");
Dialog.addString("Sample Name type: ", "Brain");
Dialog.addString("Sample Name of interest: ", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19");
Dialog.addString("Delimiter Type: ", " ")
Dialog.addString("Group Name type: ", "Slice")
Dialog.addString("Group Name of interest: ", "15,16");
Dialog.show();
Sample_Name_Type = Dialog.getString();
Sample_Name = split(Dialog.getString(), ",");
Delimiter_Type = Dialog.getString();
Group_suffix_type = Dialog.getString();
Group_suffix = split(Dialog.getString(), ","); 
ROI_Areas = "";

//Sample_Name = newArray("Brain 1", "Brain 2", "Brain 3", "Brain 4", "Brain 5", "Brain 6", "Brain 7", "Brain 8", "Brain 8", "Brain 9", "Brain 10");
//Group_suffix = newArray("Slice 1", "Slice 2", "Slice 3", "Slice 4", "Slice 5", "Slice 6", "Slice 7", "Slice 8", "Slice 9", "Slice 10", "Slice 11", "Slice 12");//must be mutually exclusive

Dialog.create("Registering Image");
Dialog.addMessage("Please specify the image width in pixel, known image width with corresponding scale for each group of image."); 
Dialog.addMessage("e.g., if you have enter 2 groups with image width in pixel 1000, 2000 before. Then put 1000, 2000 in Image known width in pixel. Similar for image known width.");
Dialog.addString("Image Known Width in pixel: ", "4606,4606");
Dialog.addString("Image Known Width: ", "9566,9566");
Dialog.addString("Image Known Width Scale: ", "um");
Dialog.show();
Group_Image_Width_In_pixel = split(Dialog.getString(), ",");
Group_Image_Known_Width = split(Dialog.getString(), ",");
Group_Image_Known_Width_Scale = Dialog.getString();

//Group_Image_Width_In_pixel = newArray("4392","4392","4392","4392","4392","4606","4606","4606","4606","4842","4842","4842");
//Group_Image_Known_Width = newArray("7537","7537","7537","7537","7537","9566","9566","9566","9566","10580","10580","10580");
//Group_Image_Known_Width_Scale = "um";

if((Group_suffix.length != Group_Image_Known_Width.length)||Group_Image_Width_In_pixel.length != Group_suffix.length){
	Dialog.create("Error: Group Number not Match");
	Dialog.addMessage("Number of Groups doesn't match with number of group width specified");
	Dialog.show();
}
for (i = 0; i < Group_suffix.length; i++) {
	print("Slice selected: "+Group_suffix[i]);
}



Dialog.create("Color Channel Suffix");
Dialog.addMessage("Put your color channel suffix here, e.g., if your image has Brain 1 Slice 1 GFP, then replace GFP with Green, or DAPI replace Blue.");
Dialog.addString("Color Channel 1 suffix: ", "green");
Dialog.addString("Color Channel 2 suffix: ", "blue");
Dialog.addString("Color Channel 3 suffix: ", "red");
Dialog.show();
Color_Channel1_suffix = Dialog.getString();
Color_Channel2_suffix = Dialog.getString();
Color_Channel3_suffix = Dialog.getString();

//Color_Channel1_suffix = "Green";
//Color_Channel2_suffix = "Blue";
//Color_Channel3_suffix = "Overlay";

Dialog.create("Particle Size range")
Dialog.addMessage("The expressed cell size range, e.g., for GFP put 1-100");
Dialog.addString("Particle Size Range: ", "10-100");
Dialog.addString("Particle Identified result show: ", "Bare Outlines");
Dialog.addString("Auto Threshold Method: ", "RenyiEntropy dark");
Dialog.show();
Particle_Size_Range = Dialog.getString();
Analyze_Particle_show = Dialog.getString();
AutoThreshold_Method = Dialog.getString();
//Particle_Size_Range = "10-100";
//Analyze_Particle_show = "Bare Outlines";

Slice_Group_Found = false;
Slice_Color_Channel_Found = false;




filelist = getFileList(directory);
//Array.sort(filelist);

/*
 * for (i = 0; i < lengthOf(filelist); i++) {
 * print(File.getName(directory + File.separator + filelist[i]));
 * }
 */

function SetROI(ROI_Directory_To_Save, imageName, ImageNameArray, Delimiter, SliceGroup){
	if(ImageNameArray[1] == Sample_Name[0]){
		waitForUser("Draw a ROI, then click OK. Adjust previous ROI to fit."); // ask user to draw the roi
    	roiManager("add");
    	selectWindow("ROI Manager");
    	roiManager("Select", 0);
    	roiManager("Rename", imageName + " ROI");
    	roiManager("Select", 0);
    	roiManager("Save", ROI_Directory_To_Save+File.separator+Sample_Name_Type+Sample_Name[0]+" "+SliceGroup+" ROI.zip");//save first slice data as reference frame from ROIManager
    	roiManager("Select", 0);
    	roiManager("Save", ROI_Directory_To_Save+File.separator+imageName+" ROI.zip");//save the ROI from ROIManager
    	roiManager("delete");//Clear ROIManager
		}
	else{
		if(isOpen("ROI Manager")){
			selectWindow("ROI Manager");
		}else{
			run("ROI Manager...");
		}
		roiManager("open", ROI_Directory_To_Save+File.separator+Sample_Name_Type+Sample_Name[0]+" "+SliceGroup+" ROI.zip");
		roiManager("Select", 0);
		waitForUser("Draw a ROI, then click OK. Adjust previous ROI to fit."); // ask user to adjust the roi
		roiManager("Select", 0);
		roiManager("update");
    	roiManager("Rename", imageName + " ROI");
    	roiManager("Select", 0);
    	roiManager("Save", ROI_Directory_To_Save+File.separator+imageName+" ROI.zip");//save the ROI from ROIManager	
    	roiManager("delete");//Clear ROIManager
	}
}


for (i = 0; i < lengthOf(filelist); i++) {
	Slice_Group_Found = false;
	Slice_Color_Channel_Found = false;
    if (endsWith(filelist[i], ".tif")) {   	
    	imageNameExt = File.getName(directory + File.separator + filelist[i]);//Get name of each image with extension in the directory
    	imageName = File.getNameWithoutExtension(directory + File.separator + filelist[i]);//get name without extension
    	imageNameArray = split(imageName, Delimiter_Type);
    	Slice_group_suffix = imageNameArray[2] +Delimiter_Type+imageNameArray[3];
    	print("Image Slice Group is: "+Slice_group_suffix);
    	directoryName = File.getName(directory);
        print("processing for "+ imageName);
        print("Number of image processed: "+ i);
        Sample_Name_Found = false;
        for (k = 0; k < Sample_Name.length; k++){
        	if(imageNameArray[1] == Sample_Name[k]){
        		Sample_Name_Found = true;
        	}
        }
        if(Sample_Name_Found){
        	for (j = 0; j < Group_suffix.length; j++) {
        	if(imageNameArray[3] == Group_suffix[j]){
        		open(directory + File.separator + filelist[i]);
        		run("Set Scale...", "distance="+Group_Image_Width_In_pixel[j]+" known="+Group_Image_Known_Width[j]+" unit="+Group_Image_Known_Width_Scale);
       			print("setting scale for "+imageName + " as "+Group_suffix[j]);
        		Slice_Group_Found = true;
        		}
        	}
        }
           
        if(Slice_Group_Found){
        	run("Split Channels");
        	if(endsWith(imageName, Color_Channel1_suffix)){//Green for GFP
        		//only select the RGB channel we want, close others
				selectWindow(imageNameExt+" (blue)");
				close();
				selectWindow(imageNameExt+" (red)");
				close();
				selectWindow(imageNameExt+" (green)");
				Slice_Color_Channel_Found = true;
				Image_RGB_Channel_Name = imageNameExt+" (green)";
        	}
        	else if(endsWith(imageName, Color_Channel2_suffix)){//blue for DAPI
        		selectWindow(imageNameExt+" (blue)");
				close();
				selectWindow(imageNameExt+" (red)");
				close();
				selectWindow(imageNameExt+" (green)");
				close();//blue channel not in use yet
				Image_RGB_Channel_Name = imageNameExt+" (blue)";
        	}else if(endsWith(imageName, Color_Channel3_suffix)){//Overlay
        		selectWindow(imageNameExt+" (blue)");
				close();
				selectWindow(imageNameExt+" (red)");
				close();
				selectWindow(imageNameExt+" (green)");
				close();//overlay channel not in use yet
				Image_RGB_Channel_Name = imageNameExt+" (red)";
        	}
        }
		
		if(Slice_Color_Channel_Found && Slice_Group_Found){
			//Set Measurements
			run("Set Measurements...", "area centroid perimeter redirect=None decimal=3");
			
			//run("Threshold...");
			setAutoThreshold(AutoThreshold_Method);
			
			print("Setting ROI for "+ Image_RGB_Channel_Name );
			SetROI(ROI_Directory_To_Save, imageName, imageNameArray, Delimiter_Type, Slice_group_suffix);
			print("Selecting ROI Manager");
			roiManager("open", ROI_Directory_To_Save+File.separator+imageName+" ROI.zip");
			roiManager("Select", 0);
			//Select the ROI from what was chosen before for each slice
			run("Measure");
			saveAs("Results", DestinationDirectory+File.separator+imageName +" ROI Area" +".csv");
			run("Clear Results");
			
			run("Analyze Particles...", "size="+Particle_Size_Range+" show=["+Analyze_Particle_show+"] display summarize add");
			selectWindow("Drawing of "+Image_RGB_Channel_Name);	
			saveAs("Tiff", DestinationDirectory + File.separator + "Drawing of "+imageName);
			close();//close the drawing of outline
			saveAs("Results", DestinationDirectory+File.separator+imageName+".csv");//save the results
			run("Clear Results");
			roiManager("Select", newArray(0));
			run("Select All");//select all points in ROImanager
			roiManager("Save", DestinationDirectory+File.separator+imageName+".zip");//save the data from ROIManager
			roiManager("delete");//Clear ROIManager
		}	
    } 
}
