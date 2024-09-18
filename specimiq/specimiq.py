import rasterio
import h5py
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import spectral
import warnings

import ipdb

class SpecimIQ():

    def __init__(self):
        self.wavelengths = np.array([397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29, 426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25, 455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32, 484.23, 487.14, 490.06, 492.97, 495.89, 498.80, 501.72, 504.64, 507.56, 510.48, 513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75, 542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12, 572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60, 601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18, 631.15, 634.11, 637.08, 640.04, 643.01, 645.98, 648.95, 651.92, 654.89, 657.87, 660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65, 690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54, 720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54, 750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64, 780.65, 783.67, 786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84, 810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07, 835.11, 838.14, 841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55, 871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95, 896.01, 899.06, 902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68, 932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40, 963.47, 966.55, 969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22, 994.31, 997.40, 1000.49, 1003.58])

    def read_envi(self, path):
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            data = src.read()

        return data

    def read_whiteref(self, path):
        if not path.endswith(".raw"):
            name = path.split("/")[-1] if "/" in path else path
            path = os.path.join(path,"capture", f"WHITEREF_{name}.raw")
            
        if os.path.exists(path):
            return self.read_envi(path)
        else:
            raise Exception("Cannot find whiteref file. \"path\" should direct to a .raw file or to the root folder of Specim IQ acquisition")
    
    def read_darkref(self, path):
        if not path.endswith(".raw"):
            name = path.split("/")[-1] if "/" in path else path
            path = os.path.join(path,"capture", f"DARKREF_{name}.raw")
            
        if os.path.exists(path):
            return self.read_envi(path)
        else:
            raise Exception("Cannot find darkref file. \"path\" should direct to a .raw file or to the root folder of Specim IQ acquisition")

    def read_radiance(self, path):
        if not path.endswith(".raw"):
            name = path.split("/")[-1] if "/" in path else path
            path = os.path.join(path,"capture", f"{name}.raw")
            
        if os.path.exists(path):
            return self.read_envi(path)
        else:
            raise Exception("Couldn't find radiance file. \"path\" should direct to a .raw file or to the root folder of Specim IQ acquisition")

    def read_rgb(self, path=None, sensor=None):
        if path != None and sensor != None:
            name = path.split("/")[-1] if "/" in path else path  
            if sensor.lower() == "spectral":
                name = f"REFLECTANCE_{name}.png"
            elif sensor.lower() == "rgb":
                name = f"RGBBACKGROUND_{name}.png"
            else:
                raise Exception("sensor type not recognized, should be \"spectral\" or \"rgb\"")
            path = os.path.join(path, "results", name)
        
        if os.path.exists(path) and path.endswith(".png"):
            img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGB).transpose(2,0,1)

            return img
        else:
            raise Exception("Couldn't find rgb file. You should specify either the direct path to the .png file or the root folder of Specim IQ acquisition")
        
    def read_reflectance(self, path, whiteref=None):
        if path.endswith(".dat"):
            if os.path.exists(path):
                return self.read_envi(path)
            else:
                raise Exception("Couldn't find reflectance file. You should specify either the direct path to the .dat file or the root folder of the Specim IQ acquisition")
        
        if whiteref == None or whiteref.lower() == "captured":
            name = path.split("/")[-1] if "/" in path else path
            path = os.path.join(path, "results" ,f"REFLECTANCE_{name}.dat")
            if os.path.exists(path):
                return self.read_envi(path)
            else:
                raise Exception("Couldn't find reflectance file. You should specify either the direct path to the .dat file or the root folder of the Specim IQ acquisition")
            
        elif whiteref.lower() == "pick":
            radiance = self.read_radiance(path)
            darkref = self.read_darkref(path)

            try:
                preview = (radiance[50]/radiance[50].max()).astype(np.float32)
                roi = cv2.selectROI("bbox", preview, showCrosshair=False, fromCenter=False)
                cv2.destroyAllWindows()
            except:
                raise Exception("Couln't perform ROI picking. Please consider using whiteref=\"captured\"")
            x,y,w,h = roi[0], roi[1], roi[2], roi[3]
            white_patch = radiance[:,y:y+h, x:x+w]
            whiteref = white_patch.mean(axis=(1,2))

            reflectance = (radiance - darkref)/(whiteref[:,None,None]-darkref)
            reflectance = reflectance.astype(np.float32)
            
            return reflectance
    
    def read_xml(self, path):
        lines = []
        with open(path, "r") as f:
            for line in f:
                lines.append(line.strip())

        return lines
    
    def xml_query(self, xml, key):
        for line in xml:
            if key in line:
                value = line.replace("</key>","").split(">")[-1]
                return value
        return None
    
    def read_metadata(self, path):
        if not path.endswith(".xml"):
            name = path.split("/")[-1] if "/" in path else path
            path = os.path.join(path,"metadata", f"{name}.xml")
            
        if os.path.exists(path):
            xml = self.read_xml(path)
            metadata = {}
            for key in ["datetime", "datacube_angle", "integration_time"]:
                metadata[key] = self.xml_query(xml, key)
            return metadata
        else:
            raise Exception("Cannot find whiteref file. \"path\" should direct to a .raw file or to the root folder of Specim IQ acquisition")


    def read(self, path):
        if not os.path.exists(path):
            raise Exception("Path to Specim IQ acquisition does not exist")
        
        result = dict(
            wavelengths = self.wavelengths,
            reflectance = self.read_reflectance(path),
            radiance = self.read_radiance(path),
            whiteref = self.read_whiteref(path),
            darkref = self.read_darkref(path),
            rgb_sensor = self.read_rgb(path, sensor="rgb"),
            simulated_rgb = self.read_rgb(path, sensor="spectral"),
            metadata = self.read_metadata(path),
            )
        
        return result
    
    def to_hf5(self, dataset, path):
        hf = h5py.File(path, "w")
        for k in dataset.keys():
            if type(dataset[k]) != dict:
                hf.create_dataset(k, data=dataset[k])
            else:
                for kk in dataset[k].keys():
                    hf.create_dataset(kk, data=dataset[k][kk])

        hf.close()

    def specim2hf5(self, specim_file, out_path):
        specim = self.read(specim_file)
        self.to_hf5(specim, out_path)
    



        

