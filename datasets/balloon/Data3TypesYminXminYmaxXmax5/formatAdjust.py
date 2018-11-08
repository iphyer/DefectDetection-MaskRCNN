#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  formatAdjust.py
#  
#  Copyright 2018 iphyer <iphyer@iphyer-HP-ProBook-4441s>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import os

from os import listdir
from os.path import isfile, join

def main(args):
	# list files in bounding boxes
	mypath = "old_bounding_boxes"
	filesList = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	# create result directory
	resultdirectory = "bounding_boxes"
	if not os.path.exists(resultdirectory):
		os.makedirs(resultdirectory)
	# loop over images in files
	for imgInfo in filesList:
		with open( mypath+"/"+imgInfo ) as imgf:
			lines = imgf.readlines()
			with open( resultdirectory+"/"+imgInfo ,"w+") as resultimgf:
				for line in lines:
					llist = line.strip().split(" ")
					resultimgf.write(llist[0] + " " + llist[2]+ " " + llist[1]+ " " + llist[4]+ " " + llist[3] + "\n")
	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
