## codes to extract tables from pdfs : contextual to a specific personal problem. beginning codes are the primary 


# Snippet for extracting info from the pdfs

{
install.packages("tesseract")
install.packages("pdftables")
ghit::install_github(c("leeper/tabulizerjars", "leeper/tabulizer"), INSTALL_opts = "--no-multiarch", dependencies = c("Depends", "Imports"))

{
# Here are a few methods for getting text from PDF files. Do read through
# the instructions carefully! Note that this code is written for Windows 7,
# slight adjustments may be needed for other OSs

# Tell R what folder contains your 1000s of PDFs
dest <- "C:/Users/varun.v1/Documents/R/CDOT"

# make a vector of PDF file names
myfiles <- list.files(path = dest, pattern = ".pdf",  full.names = TRUE)

# now there are a few options...

############### PDF (image of text format) to TXT ##########
# This is for is your PDF is an image of text, this is the case
# if you open the PDF in a PDF viewer and you cannot select
# words or lines with your cursor.

##### Wait! #####
# Before proceeding, make sure you have a copy of Tesseract
# on your computer! Details & download:
# https://code.google.com/p/tesseract-ocr/
# and a copy of ImageMagick: http://www.imagemagick.org/
# and a copy of pdftoppm on your computer!
# Download: http://www.foolabs.com/xpdf/download.html
# And then after installing those three, restart to
# ensure R can find them on your path.
# And note that this process can be quite slow...

# PDF filenames can't have spaces in them for these operations
# so let's get rid of the spaces in the filenames

sapply(myfiles, FUN = function(i){
  file.rename(from = i, to =  paste0(dirname(i), "/", gsub(" ", "", basename(i))))
})

# get the PDF file names without spaces
myfiles <- list.files(path = dest, pattern = ".pdf",  full.names = TRUE)

# Now we can do the OCR to the renamed PDF files. Don't worry
# if you get messages like 'Config Error: No display
# font for...' it's nothing to worry about

lapply(myfiles, function(i){
  # convert pdf to ppm (an image format), just pages 1-10 of the PDF
  # but you can change that easily, just remove or edit the
  # -f 1 -l 10 bit in the line below
  shell(shQuote(paste0("pdftoppm ", i, " -f 1 -l 10 -r 600 ocrbook")))
  # convert ppm to tif ready for tesseract
  shell(shQuote(paste0("convert *.ppm ", i, ".tif")))
  # convert tif to text file
  shell(shQuote(paste0("tesseract ", i, ".tif ", i, " -l eng")))
  # delete tif file
  file.remove(paste0(i, ".tif" ))
})


# where are the txt files you just made?
dest # in this folder

# And now you're ready to do some text mining on the text files

############### PDF (text format) to TXT ###################

##### Wait! #####
# Before proceeding, make sure you have a copy of pdf2text
# on your computer! Details: https://en.wikipedia.org/wiki/Pdftotext
# Download: http://www.foolabs.com/xpdf/download.html

# If you have a PDF with text, ie you can open the PDF in a
# PDF viewer and select text with your curser, then use these
# lines to convert each PDF file that is named in the vector
# into text file is created in the same directory as the PDFs
# note that my pdftotext.exe is in a different location to yours
lapply(myfiles, function(i) system(paste('"C:/Users/varun.v1/Documents/R/CDOT/pdftotext.exe"',
                                         paste0('"', i, '"')), wait = FALSE) )

# where are the txt files you just made?
dest # in this folder

# And now you're ready to do some text mining on the text files

############### PDF to CSV (DfR format) ####################

# or if you want DFR-style csv files...
# read txt files into R
mytxtfiles <- list.files(path = dest, pattern = "txt",  full.names = TRUE)

library(tm)
mycorpus <- Corpus(DirSource(dest, pattern = "txt"))
# warnings may appear after you run the previous line, they
# can be ignored
mycorpus <- tm_map(mycorpus,  removeNumbers)
mycorpus <- tm_map(mycorpus,  removePunctuation)
mycorpus <- tm_map(mycorpus,  stripWhitespace)
mydtm <- DocumentTermMatrix(mycorpus)
# remove some OCR weirdness
# words with more than 2 consecutive characters
mydtm <- mydtm[,!grepl("(.)\\1{2,}", mydtm$dimnames$Terms)]

# get each doc as a csv with words and counts
for(i in 1:nrow(mydtm)){
  # get word counts
  counts <- as.vector(as.matrix(mydtm[1,]))
  # get words
  words <- mydtm$dimnames$Terms
  # combine into data frame
  df <- data.frame(word = words, count = counts,stringsAsFactors = FALSE)
  # exclude words with count of zero
  df <- df[df$count != 0,]
  # write to CSV with original txt filename
  write.csv(df, paste0(mydtm$dimnames$Docs[i],".csv"), row.names = FALSE)
}

# and now you're ready to work with the csv files

############### PDF to TXT (all text between two words) ####

## Below is about splitting the text files at certain characters
## can be skipped...

# if you just want the abstracts, we can use regex to extract that part of
# each txt file, Assumes that the abstract is always between the words 'Abstract'
# and 'Introduction'

abstracts <- lapply(mytxtfiles, function(i) {
  j <- paste0(scan(i, what = character()), collapse = " ")
  regmatches(j, gregexpr("(?<=Abstract).*?(?=Introduction)", j, perl=TRUE))
})
# Write abstracts into separate txt files...

# write abstracts as txt files
# (or use them in the list for whatever you want to do next)
lapply(1:length(abstracts),  function(i) write.table(abstracts[i], file=paste(mytxtfiles[i], "abstract", "txt", sep="."), quote = FALSE, row.names = FALSE, col.names = FALSE, eol = " " ))

# And now you're ready to do some text mining on the txt

# originally on http://stackoverflow.com/a/21449040/1036500
}

library(tabulizer)
# library(dplyr)
tables = extract_tables('*.pdf', method = "data.frame")


# below method is good
library(pdftables)
# library(readxl)
# need to register in the pdftables webiste and generate an api key for each instance. not sure on when it expires.
convert_pdf(input_file = '*.pdf', output_file = 'out.xlsx', format = 'xlsx-multiple', api_key = "hrxt7bhsj62s")

# cmd lines
# apt-get install libcurl4-openssl-dev
}

# Reading in the data, cleaning, appending to the other file, exporting the final file with the required columns
{

# table
{
table_1 = data.frame(read_excel(path = "C:/Users/varun.v1/Documents/R/CDOT/out.xlsx", sheet = 1, skip = 2, col_names = T))
table_2 = data.frame(read_excel(path = "C:/Users/varun.v1/Documents/R/CDOT/out.xlsx", sheet = 2, skip = 2, col_names = T))
table_3 = data.frame(read_excel(path = "C:/Users/varun.v1/Documents/R/CDOT/out.xlsx", sheet = 3, skip = 2, col_names = T))
table_4 = data.frame(read_excel(path = "C:/Users/varun.v1/Documents/R/CDOT/out.xlsx", sheet = 4, skip = 2, col_names = T))

table_1[is.na(table_1)] <- ""
table_2[is.na(table_2)] <- ""
table_3[is.na(table_3)] <- ""
table_4[is.na(table_4)] <- ""

table_1$STIP.WBS.ID.Description = paste(table_1$STIP.WBS.ID.Description, table_1$NA., sep = " ")
table_1 = table_1[1:77,c(1:5,7:10)]
table_2 = table_2[1:70,]
table_3$STIP.Description = paste(table_3$STIP.Description, table_3$NA., table_3$NA..1, sep = " ")
table_3$STIP.WBS.ID.Description = paste(table_3$STIP.WBS.ID.Description, table_3$NA..2, sep = " ")
table_3 = table_3[1:71, c(1:3,6:7,9:12)]
table_4 = table_4[1:24,c(1:5,7:10)]

table = rbind(table_1, table_2, table_3, table_4)

table = table %>%
  mutate(Funding.Program = "") %>%
  mutate(Fund.Source = "") %>%
  mutate(Fund.Type = "") %>%
  mutate(STIP.Phase = "") %>%
  mutate(X2017 = "") %>%
  mutate(X2018 = "") %>%
  mutate(X2019 = "") %>%
  mutate(X2020 = "") %>%
  mutate(Future = "")
}

# tabble
{
tabble = read.csv('data.csv', header = T, as.is = T, na.strings = "")

library(zoo)
tabble = na.locf(tabble, na.rm = F)

tabble = tabble %>%
  mutate(FY16...FY19.STIP.Amount = "") %>%
  mutate(FY16...FY19.Budgeted.Amount = "") %>%
  mutate(FY2020.STIP.Amount = "") %>%
  mutate(Status = "")
}

library(data.table)

setnames(table, old = c("STIP.Description"), new = c("STIP.ID.Description"))
setnames(tabble, old = c("STIP.WBS.Description"), new = c("STIP.WBS.ID.Description"))

main_table = bind_rows(table,tabble)

Fund.table = main_table %>%
  select(STIP.WBS.ID.Description, Funding.Program, Fund.Type, Fund.Source) %>%
  unique()

main_table = main_table %>%
  select(-Funding.Program, -Fund.Source, -Fund.Type) %>%
  unique() %>%
  mutate(Status = if_else(Status == "", "Not Completed", Status)) %>%
  mutate(Perc.Completion = if_else(Status == "Not Completed", sample(c(0,25,50,75,90), n(),
                    replace = TRUE, prob = c(0.7,0.05,0.05,0.1,0.1)), 0)) %>%
  # break
  mutate(ROI.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(Monetary.Benefit.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(CDOT.Savings.Year = sample(10000:999999, n(), replace = TRUE)) %>%
  mutate(User.Savings.Year = sample(10000:999999, n(), replace = TRUE)) %>%
  mutate(Non.Monetary.Benefit.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(Jobs.Created = sample(100:9999, n(), replace = TRUE)) %>%
  mutate(Job.Accessibility = sample(c("Yes","No"), n(), replace = TRUE, prob = c(0.3, 0.7))) %>%
  mutate(Freight.Improvement = sample(c("Yes","No"), n(), replace = TRUE, prob = c(0.05, 0.95))) %>%
  mutate(Environment.Improvement = sample(c("Yes","No"), n(), replace = TRUE, prob = c(0.1, 0.9))) %>%
  mutate(Aesthetics.Improvement = sample(c("Yes","No"), n(), replace = TRUE, prob = c(0.08, 0.92))) %>%
  mutate(Tourism.Flag = if_else(Aesthetics.Improvement == "Yes",
                                sample(c("Yes","No"), n(), replace = TRUE, prob = c(0.6, 0.4)), "No")) %>%
  # break
  mutate(Multi.Modal.Project = sample(c("Yes","No"), n(), replace = TRUE, prob = c(0.05, 0.95))) %>%
  mutate(Modes.Concerned = if_else(Multi.Modal.Project == "Yes",
                                   sample(c(2,3,4,5), n(), replace = TRUE, prob = c(0.5,0.3,0.15,0.05)), 1)) %>%
  mutate(Mode.Transportation = "") %>%
  # break
  mutate(Risk.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(Technical.Risk.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(Management.Risk.Index = sample(1:10, n(), replace = TRUE)) %>%
  # break
  mutate(Impact.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(Environmental.Impact.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(Social.Impact.Index = sample(1:10, n(), replace = TRUE)) %>%
  # break
  mutate(Effort.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(Right.Of.Way.Flag = sample(c("Yes","No"), n(), replace = TRUE, prob = c(0.3, 0.7))) %>%
  mutate(Time.Completion = sample(1:5, n(), replace = TRUE, prob = c(0.3,0.2,0.2,0.2,0.1))) %>%
  mutate(Manpower = sample(c("High","Medium","Low"), n(), replace = TRUE, prob = c(0.2,0.5,0.3))) %>%
  mutate(InterAgency.Collaboration.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(Agencies.Involved = sample(2:15, n(), replace = TRUE)) %>%
  mutate(Federal.Agencies = sample(c("Yes","No"), n(), replace = TRUE, prob = c(0.05, 0.95))) %>%
  mutate(Vendors = sample(c("Yes","No"), n(), replace = TRUE, prob = c(0.02, 0.98))) %>%
  # break
  mutate(Performance.Measures.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(Safety.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(Mobility.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(Livability.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(PTI.Improvement.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(Infrastructure.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(Maintenance.LOS.Index = sample(1:10, n(), replace = TRUE)) %>%
  # break
  mutate(Emergency.Flag = sample(c("Yes","No"), n(), replace = TRUE, prob = c(0.03, 0.97))) %>%
  mutate(Compliance = sample(c("High","Medium","Low"), n(), replace = TRUE, prob = c(0.2,0.5,0.3))) %>%
  mutate(Project.Complexity.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(Ease.Of.Construction = sample(c("Yes","No"), n(), replace = TRUE, prob = c(0.05, 0.95))) %>%
  mutate(Ease.Of.Funding = sample(c("Yes","No"), n(), replace = TRUE, prob = c(0.05, 0.95))) %>%
  mutate(Ease.Of.Design = sample(c("Yes","No"), n(), replace = TRUE, prob = c(0.05, 0.95))) %>%
  # break
  mutate(Concerned.Goal = sample(c("Safety","Mobility","Maintaining the system","Economic Vitality"), n(),
                               replace = TRUE, prob = c(0.2, 0.2, 0.5, 0.1)))

main_table_1 = main_table %>%
  select(CDOT.Region) %>%
  unique() %>%
  mutate(Region.Priority.Index = sample(1:10, n(), replace = TRUE)) %>%
  mutate(Population = sample(10000:50000, n(), replace = TRUE)) %>%
  mutate(Area.Covered = sample(10000:50000, n(), replace = TRUE)) %>%
  mutate(Geographical.Feasibility.Index = sample(1:10, n(), replace = TRUE))
  # break

main_table = inner_join(main_table, main_table_1)

main_table = main_table %>%
  mutate(Priority.Score = (ROI.Index/Effort.Index)*10 + Impact.Index + Risk.Index
         + Performance.Measures.Index + if_else(Emergency.Flag == "Yes", 5, 0)
         + if_else(Compliance == "High", 3, if_else(Compliance == "Medium", 2, 1))
         + (Project.Complexity.Index/5) + (Region.Priority.Index/2))


main_table = inner_join(main_table, Fund.table)
}

# Export the table
write.csv(main_table, "CDOT_Prioritization_Framework.csv", row.names = F)
