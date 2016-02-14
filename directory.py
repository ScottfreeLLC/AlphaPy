##############################################################
#
# Package   : AlphaPy
# Module    : directory
# Version   : 1.0
# Copyright : Mark Conway
# Date      : June 29, 2013
#
##############################################################


#
# Function dirname
#

def dirname(name, subject):
	return '_'.join([name, subject + 's'])


#
# Class Directory
#

class Directory(object):
	
	# frame list
	
	directories = {}
	
	# __init__
	
	def __init__(self,
				 name,
				 subject,
				 uri,
				 sep,
				 key):
		# code
			dn = dirname(name, subject)
			if not dn in Directory.directories:
				self.name = name
				self.subject = subject
				self.uri = uri
				self.sep = sep
				self.key = key
				# add directory to directory list
				Directory.directories[dn] = self
			else:
				print "Directory %s already exists" % dn
		
	# __str__

	def __str__(self):
		return dirname(self.name, self.subject)


"""
splatr.newdirectory <-
	function(name = "nasdaq",
			   subject = "stock",
			   uri = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt",
			   sep = "|",
			   key = "Symbol")
{
	dname <- ""
	newdir <- new("splatr.directory",
				    name = name,
				    subject = subject,
				    uri = uri,
				    sep = sep,
				    key = key)
	if (!is.null(newdir)) {
		newdir$frame <- read.csv(uri, sep=sep)
      dname <- splatr.getdirectoryname(name, subject)
		splatr.setdirectory(dname, newdir)
	}
	dname
}

splatr.dlookup <- function(dir, keyvalue)
{
	dirf <- dir$frame
	with(dirf, dirf[get(dir$key) == keyvalue, ])
}
"""