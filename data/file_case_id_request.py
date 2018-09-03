import requests
import json
import pandas as pd
import os

def retrieveFileMeta(file_ids,outputfile):
    '''

    Get the tsv metadata for the list of case_ids
    Args:
        file_ids: an array of file_ids
        outputfile: the output filename

    '''
    fd = open(outputfile,'w')
    cases_endpt = 'https://api.gdc.cancer.gov/files'

    # The 'fields' parameter is passed as a comma-separated string of single names.
    fields = [
        "file_id",
        "file_name",
        "cases.submitter_id",
        "cases.case_id",
        "data_category",
        "data_type",
        "cases.samples.tumor_descriptor",
        "cases.samples.tissue_type",
        "cases.samples.sample_type",
        "cases.samples.submitter_id",
        "cases.samples.sample_id",
        "cases.samples.portions.analytes.aliquots.aliquot_id",
        "cases.samples.portions.analytes.aliquots.submitter_id",
        ]

    filters = { 
        "op":"in",
        "content":{
            "field": "files.file_id",
            "value" : file_ids.tolist()
        }
        
    }
    print(filters)
    fields = ','.join(fields)

    params = {
        "filter" : filters,
        "fields": fields,
        "format": "TSV",
        "pretty": "true",
        }
    print (filters)
    print (fields)
    response = requests.post(cases_endpt, headers = {"Content-Type": "application/json"},json = params)

    fd.write(response.content.decode("utf-8"))
    fd.close()

    # print(response.content)


def genPayload(file_ids,payloadfile):
    '''
    Used for the curl method to generate the payload.
    '''
    fd = open(payloadfile,"w")
    filters = {
        "filters":{
            "op":"in",
            "content":{
                "field":"files.file_id",
                "value": file_ids.tolist()
            }
        },
        "format":"TSV",
        "fields":"file_id,file_name,cases.submitter_id,cases.case_id,data_category,data_type,cases.samples.tumor_descriptor,cases.samples.tissue_type,cases.samples.sample_type,cases.samples.submitter_id,cases.samples.sample_id,cases.samples.portions.analytes.aliquots.aliquot_id,cases.samples.portions.analytes.aliquots.submitter_id",
        "pretty":"true"
    }
    json_str = json.dumps(filters)
    fd.write(json_str)
    fd.close()
    # return json_str

def getFileids(filename):
    df = pd.read_csv(filename)
    file_ids = df.file_id.values
    return file_ids

if __name__ == '__main__':

    filename = "file_case_id.csv"
    
    

    file_ids = getFileids(filename)
    # print (file_ids)

    fileids_meta_outfile = "files_meta.tsv"
    # request method
    retrieveFileMeta(file_ids,fileids_meta_outfile)


    # the curl method
    
    payloadfile = "Payload"
    genPayload(file_ids,payloadfile)
    os.system("curl --request POST --header \"Content-Type: application/json\" --data @Payload 'https://api.gdc.cancer.gov/files' > File_metadata.txt")
    
