import urllib.request
import os
import sys
import glob
import json
import datetime
import codecs

train_url = 'http://planktivore.ucsd.edu/data/rois/images/SPCP2/1510396800000/1537683199000/0/24/1000/20/2000/0.0/1.0/clipped/ordered/skip/Ceratium furca/anytype/Any/Any/'

testset01_url = 'http://planktivore.ucsd.edu/data/rois/images/SPCP2/1522108800000/1530403200000/0/24/1000/40/1000/0.0/1.0/clipped/randomize/skip/Any/anytype/Any/Any/'

testset02_url = 'http://planktivore.ucsd.edu/data/rois/images/SPCP2/1514764800000/1514851200000/0/24/10000/50/1000/0.0/1.0/clipped/randomize/skip/Any/anytype/Any/Any/'

class SPCQueryURL:

    param_names = [
        'base_url',
        'camera',
        'start_utc',
        'end_utc',
        'start_hour',
        'end_hour',
        'n_images',
        'min_maj',
        'max_maj',
        'min_aspect',
        'max_aspect',
        'clipped',
        'ordered',
        'make_archive',
        'label',
        'label_type',
        'tag',
        'annotator'
    ]

    def __init__(self,query_params={}):
        self.query_params = query_params

    def set_from_url(self,input_url):
        url_tokens = input_url.split('/')
        self.query_params['base_url']  = 'http://' + url_tokens[2] + '/' + url_tokens[3] + '/' + url_tokens[4] + '/' + url_tokens[5]
        param_index = 1
        for t in url_tokens[6:]:
            if len(t) > 0:
                self.query_params[self.param_names[param_index]] = t
            param_index += 1

        print(self.query_params)

    def increment_day(self, num_days=1):
        """
        Increments the query url by the appropriate number of seconds
        :param num_days: multiplier for number of days [int]
        """

        self.query_params['start_utc'] = str(int(self.query_params['start_utc']) + 24*3600*1000*num_days)
        self.query_params['end_utc'] = str(int(self.query_params['end_utc']) + 24*3600*1000*num_days)

    def get_url(self):

        url = ''

        for t in self.param_names:
            url = url + self.query_params[t] + '/'

        return url

    def set_label(self,new_label):
        self.query_params['label'] = new_label

    def time_range(self,start_time,end_time):
        try:
            self.query_params['start_time'] = str(int(start_time)*1000)
            self.query_params['end_time'] = str(int(start_time)*1000)
        except:
            print('Please format start and end times as unixtime ints or strings.')

    def size_range(self,min_maj,max_maj,min_aspect,max_aspect):

        try:
            self.query_params['min_maj'] = str(min_maj)
            self.query_params['max_maj'] = str(max_maj)
            self.query_params['min_aspect'] = str(min_aspect)
            self.query_params['min_aspect'] = str(max_aspect)
        except:
            print('Error parsing size range, input should be floats or strings')



def get_json_data(url):

    response = urllib.request.urlopen(url)
    reader = codecs.getreader("utf-8")
    data = json.load(reader(response))

    return data

if __name__=='__main__':

    query = SPCQueryURL()
    query.set_from_url(testset02_url)

    query.set_label(sys.argv[2])

    output_dir = os.path.join(sys.argv[1],sys.argv[2])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(sys.argv) > 4:
        iterations = int(sys.argv[4])
    else:
        iterations = 1

    if len(sys.argv) > 5:
        total_days = int(sys.argv[5])
    else:
        total_days = 1

    for i in range(0,iterations):


        print('Data Download Iteration ' + str(i))

        query.increment_day()

        for day in range(0,total_days):

            print('Day: ' + query.query_params['start_utc'])

            output_subdir = os.path.join(output_dir,query.query_params['start_utc'],'images')
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            else:
                print('skipping dir as it already exists...')
                query.increment_day()
                continue

            # set the start and end Day
            print(query.get_url())
            data = get_json_data(query.get_url())
            data = data['image_data']



            for im in data['results']:
                try:
                    image_url = sys.argv[3] + '/' + im['image_url'] + '.jpg'
                    output_file = os.path.join(output_subdir,im['image_id'][:-4] + '.jpg')
                    print(output_file)
                    urllib.request.urlretrieve(image_url,output_file)
                except:
                    print('error in downloading image')

            query.increment_day()
