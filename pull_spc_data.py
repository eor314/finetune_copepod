# -*- coding: utf-8 -*-

"""
Created on Thur May 10, 2018

Downloads all images in a date and size range from SPC server. Datetime formatting mimics input from front end
including timezone aware objects.

run from command line:

$ python pull_spc_data.py startTime endTime out_folder

with startTime and endTime as 'YYYY-MM-DD'

"""

import urllib3
import os
import datetime
import pytz
import calendar
import json
import numpy as np
from sys import argv


def convert_date(start_time, end_time):

    """
    Convert datetime object to timestamp
    :param start_time: start time YYYY-MM-DD HH:MM:SS
    :param end_time: end time YYYY-MM-DD HH:MM:SS
    :return: start and end timestamp in html format
    """
    # convert date time string in Pacific time to unix time stamp.startTime and endTime are strings in 'YYYY-MM-DD HH:MM:SS'
    start_ob = pytz.timezone('America/Los_Angeles').localize(datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S'))
    end_ob = pytz.timezone('America/Los_Angeles').localize(datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S'))

    utc_start = calendar.timegm(start_ob.utctimetuple())
    utc_end = calendar.timegm(end_ob.utctimetuple())

    # Add 3600 to for daylight savings time.
    if start_ob.timetuple().tm_isdst == 1:
        utc_start = utc_start + 3600

    if end_ob.timetuple().tm_isdst == 1:
        utc_end = utc_end + 3600

    # don't forget to multiply by 1000 to make the utc format into microseconds for app/home/eric/python/oithona_classificationroriate http plug in
    utc_start = utc_start * 1000
    utc_end = utc_end * 1000

    # also add the original datetime object to determine server location
    return utc_start, utc_end, start_ob


# the url builder
def url_builder(start, end, mn, mx):

    """
    Builds the url to query SPC database from the front end

    :param start: start timestamp in html format
    :param end: end timestamp in html format
    :param mn: minimum length (mm)
    :param mx: maximum length (mm)
    :return: query url
    """

    cam_res = 7.38 / 1000  # resolution of SPC2

    min_l = np.floor(float(mn) / cam_res)
    max_l = np.ceil(float(mx) / cam_res)

    # convert datetime
    utcs, utce, s_ob = convert_date(start, end)

    # set server location based on the date
    if s_ob.date() < datetime.date(2017, 7, 31):
        pattern = "http://spc.ucsd.edu/data/rois/images/SPC2/{!s}/{!s}/0/24/300/{!s}/{!s}/0.05/1/noexclude/ordered/skip/Any/anytype/Any/Any"
    else:
        pattern = "http://planktivore.ucsd.edu/data/rois/images/SPC2/{!s}/{!s}/0/24/300/{!s}/{!s}/0.05/1/noexclude/ordered/skip/Any/anytype/Any/Any"

    out = pattern.format(utcs, utce, int(min_l), int(max_l))
    return out


def download_spc_url(url_string, out_path):

    """
    Open an SPC database url and iterate over each page

    :param url_string: URL to spc database location (str)
    :param out_path: location to download images (str)
    :return:
    """

    # set up the next page and appropriate strings
    next_page = 'empty'

    # start the pool manager
    http = urllib3.PoolManager()

    page_base = url_string.split('.')[0]
    inurl = page_base + '.ucsd.edu{!s}'
    imgurl = page_base + '.ucsd.edu{!s}.jpg'
    out_base = os.path.join(out_path, '{!s}')

    url_open = url_string
    req = http.request('GET', url_open)
    json_doc = json.loads(req.data.decode('utf-8'))

    # skip the url if there is no data
    if json_doc['image_data']['count'] == 0:
        pass
    else:
        # create the output file
        os.mkdir(out_string)

        while next_page:

            req = http.request('GET', url_open)
            json_doc = json.load(req.data.decode('utf-8'))

            next_page = json_doc['image_data']['next']

            # update the URL for next iteration (use index 21 to replace the internal server address with url)
            if next_page:
                url_open = inurl.format(next_page[21::])
            else:
                pass

            img_dicts = json_doc['image_data']['results']

            for ii in range(len(img_dicts)):

                # read in image url
                img_url = img_dicts[ii]['image_url']

                # make the url to download the image
                inpath = imgurl.format(img_url)

                # make the out_path
                imgout = out_base.format(os.path.basename(img_url))
    """
                # download the image. Skip if the image is not found
                try:
                    img_req = http.request('GET', inpath)
                    urlretrieve(inpath, imgout)
                except IOError:
                    pass


if __name__ == '__main__':

    with open('/media/storage/learning_files/oithona_unlabeled_days_format.txt', 'r') as ff:
        to_process = list(ff)
        ff.close()

    to_process = [line.strip() for line in to_process]
    to_process = [[line.split(' ')[0] + ' ' + line.split(' ')[1], line.split(' ')[2] +
                   ' ' + line.split(' ')[3]] for line in to_process]

    # iterate through and download images
    for item in to_process:

        # make the url
        to_open = url_builder(item[0], item[1], 0.5, 2.5)

        # make the directory for the new day of data
        out_string = os.path.join(argv[3], str(item[2]))

        # loop download the images
        download_spc_url(to_open, out_string)

        print("Done with " + item[0].split(' ')[0])
