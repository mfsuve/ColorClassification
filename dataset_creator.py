# import os
# import cv2
#
# main_folder = 'images'
# # color = ['black']
#
# try:
# 	color
# except NameError:
# 	l = os.listdir(main_folder)
# else:
# 	l = color
# for folder in l:
# 	print('Looking in {} folder'.format(folder))
# 	folder_path = os.path.join(main_folder, folder)
# 	for image in os.listdir(folder_path):
# 		image_path = os.path.join(folder_path, image)
# 		if len(image) <= 4 or image[-3:] != 'png':
# 			print('Removing', image_path)
# 			os.remove(image_path)
# 		else:
# 			try:
# 				print('Reanaming', image_path)
# 				os.rename(image_path, '{}.{}'.format(image_path[:-4], image_path[-3:]))
# 			except FileExistsError:
# 				print('Can\'t rename, Removing', image_path)
# 				os.remove(image_path)
#
# if len(color) > 0:
# 	l = color
# else:
# 	os.listdir(main_folder)
# for folder in l:
# 	print('Looking in {} folder'.format(folder))
# 	folder_path = os.path.join(main_folder, folder)
# 	for image in os.listdir(folder_path):
# 		image_path = os.path.join(folder_path, image)
# 		img = cv2.imread(image_path)
# 		if img is None:
# 			print('None Found, Removing', image_path)
# 			os.remove(image_path)

# from requests import get
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import requests
import shutil
import os

colors = ['white', 'black', 'red', 'orange', 'yellow', 'green', 'blue', 'purple']

driver_path = 'C:/Users/mus-k/Desktop/Üniversite/Image Processing/Homework2/chromedriver_win32/chromedriver.exe'
browser = webdriver.Chrome(executable_path=driver_path)
browser.maximize_window()


def get_images(color, count=100):
	browser.get('https://www.pexels.com/search/{}/'.format(color))
	print('Scrolling...')
	for _ in range(10):
		browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
		time.sleep(3)
	index = 0
	print('Creating images/{}'.format(color))
	os.makedirs('images/{}'.format(color))
	articles = browser.find_elements_by_tag_name('article')
	print('Found {} images online'.format(len(articles)))
	for article in articles:
		html = article.get_attribute('innerHTML')
		soup = BeautifulSoup(html, 'html.parser')
		images = soup.find_all('img')
		for image in images:
			src = image['srcset']
			r = requests.get(src, stream=True, headers={'User-agent': 'Mozilla/5.0'})
			if r.status_code == 200:
				with open('images/{}/image{:03d}.png'.format(color, index), 'wb') as f:
					if index % 50 == 0:
						print('Saving: image{:03d}:{:03d}.png'.format(index, min(index+50, count)))
					r.raw.decode_content = True
					shutil.copyfileobj(r.raw, f)
					index += 1
					if index >= count:
						print('Saved all. Exitting...')
						time.sleep(3)
						browser.close()
						return
	time.sleep(3)
	browser.close()


for color in colors:
	print('Running for color {}'.format(color))
	if not os.path.exists('images/{}'.format(color)):
		get_images(color, count=122)
	else:
		# raise FileExistsError
		print('Found images/{} folder'.format(color), end=' ')
		count = len(os.listdir('images/{}'.format(color)))
		print('\t-->\t{} files'.format(count))
		if count < 100:
			print('Completing the remaining {} images'.format(100 - count))
			get_images(color, 100 - count)
	time.sleep(3)

browser.quit()
print('Done image gathering')