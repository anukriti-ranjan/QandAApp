import os
import zipfile
import docx
import pandas as pd

def parse_docx_into_list(document):
      # Initialize variables to keep track of the current section and its contents
    current_section = ''
    section_contents = []
    sections = dict()

    # Loop through the paragraphs in the document
    for paragraph in document.paragraphs:
        # Check if the paragraph is a heading
        style = paragraph.style.name
        if style.startswith('Heading'):
            # If this is a new section, add the previous section to a dictionary
            if current_section != '':
                sections[(current_section)] = '\n'.join(section_contents)
                section_contents = []
            # Set the current section to the heading text without numbers or tabs
            heading_words = paragraph.text.split()
            heading_text = ' '.join(word for word in heading_words if not word[0].isdigit() and not word.startswith('\t'))
            current_section = heading_text.strip()
        else:
            # If this is not a heading, add the paragraph text to the current section's contents
            section_words = paragraph.text.split()
            section_text = ' '.join(section_words)
            section_contents.append(section_text)

    # Add the last section to the dictionary
    sections[current_section] = '\n'.join(section_contents)

    my_doc = []
    my_dict = {}
    for key, value in sections.items():
      if 'Foreword' in key: 
        pass
      elif 'References' in key:
        pass
      elif 'Definitions and abbreviations' in key:
        pass
      elif value=='':
        pass
      else:
        my_string = key + ": "+ value
        my_doc.append(my_string)
        #my_dict[key] = value

    return my_doc


# path to the zipped folder
zip_path = '/data/all_425_refs_docx.zip'
# get the base name of the file
zip_basename = os.path.basename(zip_path)

# split the name and extension
zip_name, zip_ext = os.path.splitext(zip_basename)

# path to the directory where the unzipped files will be stored
unzip_path = '/data'

# create the directory if it doesn't exist
if not os.path.exists(unzip_path):
    os.makedirs(unzip_path)

# extract the files from the zip folder to the unzipped folder
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_path)

unzipped_folder_path = os.path.join(unzip_path, zip_name)
print(unzipped_folder_path)

document_list = []
#dict_of_docs = {}

# loop through the unzipped folder and read each .docx file
for filename in os.listdir(unzipped_folder_path):
    print(filename)
    if filename.endswith('.docx'):
        doc_path = os.path.join(unzipped_folder_path, filename)
        print(doc_path)
        doc = docx.Document(doc_path)
        # do something with the document here
        print(f"Reading {doc_path} ...")
        document_list.extend(parse_docx_into_list(doc))
        #dict_of_docs.update(parse_docx_into_list(doc, filename))


def concatenate_strings(strings):
    concatenated_strings = []
    current_string = ""
    for string in strings:
        if len(current_string + string) <= 1500:
            current_string += string
        else:
            concatenated_strings.append(current_string)
            current_string = string
    concatenated_strings.append(current_string)
    return concatenated_strings

new_document_list = concatenate_strings(document_list)

my_df = pd.DataFrame({"content": new_document_list})
my_df.to_csv("data/lte_docs_segmented.csv", index=False)