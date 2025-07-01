#!/usr/bin/env python
#
# Script to deactivate tilts from Warp TS mrc-xml based on excludelist from batch etomo com files. 
# Useful to remove undesdired tilts from tomo and subtomo reconstructions.
# Tested for Warp 1.1.0-beta
#
# This script will:
# - read mrc-xmls from a back-up folder (3D-CTFs already processed, handedness checked)
# - read the excludelist from latest batch etomo aligncom file
# - edit the mrc-xml and set excluded tilts to False
# - re-write mrc-xml in original frames location
# 
# jh oct 2023
# Modified for warp linux version by SYC Sep 2024

import xml.etree.ElementTree as ET
import numpy as np
import glob
import re
import logging

def run_xml_parsing():
    """
    Main function to parse and edit MRC-XML files based on eTomo excludelists.
    This function assumes it is run from the 'warp_tiltseries/tiltstack' directory.
    """
    mrcxmls = sorted(glob.glob('../*.xml')) 

    ####################################
    ####### FUNCTION BLOCK BELOW #######

    # GET EXCLUDELIST AS ARRAY 
    def get_excludelist(aligncom):
        
        global excludelist_str
        excludelist_str = None
        
        with open(aligncom, 'r') as file:
            data = file.readlines()
        
        for line in data:
                if line.startswith('ExcludeList'):
                    excludelist_str = line.split()[1].split(',')
        
        if excludelist_str:
            excludelist = np.array([int(i) for i in excludelist_str])
            logging.info(f"Tilts to exclude are {excludelist}")
        else:
            logging.info("No tilts to exclude")
            excludelist = np.array([])
            
        return excludelist

    # PARSE AND EDIT MRC-XML
    def edit_mrcxml_usetilts(path_to_xml,excludelist,output_xml_basename):
        
        # parse XML, find UseTilt list
        tree = ET.parse(path_to_xml)
        root = tree.getroot()
        a = root.findall('UseTilt')
        
        if excludelist.any():
            logging.info("Found tilts to exclude. Modifying XML.")

            # edit excluded tilts
            excludelist = excludelist - 1 # account for 0 index
            usetilts = a[0].text # list boolean as single string
            usetilts_mod = usetilts.split()
            for i in excludelist:
                usetilts_mod[i] = 'False'
            usetilts_new = '\n'.join(usetilts_mod)

            #print(usetilts_new)
            a[0].text = usetilts_new
            root.set('UseTilt',usetilts_new)
            tree.write(f'{output_xml_basename}.xml')
            
        else:
            logging.info("No tilts to exclude in this file.")
            tree.write(f'{output_xml_basename}.xml')


    ####### FUNCTION BLOCK ABOVE #######
    ####################################

    for i in mrcxmls:
        
        logging.info(f'Processing {i}...')
        
        # get basename
        ts_base = re.split('.xml|/',i)[-2]
        
        aligncom = f'{ts_base}/align_clean.com'
        try:
            excludelist = get_excludelist(aligncom)
        except FileNotFoundError:
            logging.warning(f"{aligncom} not found, skipping exclusion list.")
            excludelist = np.array([])
        
        # run mrcxml editing
        output_xml_basename = f'../{ts_base}'
        edit_mrcxml_usetilts(i,excludelist,output_xml_basename)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    run_xml_parsing()
