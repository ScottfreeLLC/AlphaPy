Quick Start
===========

Crawl a web page

The most simple way to use our program is with no arguments.
Simply run:

python main.py -u <url>

to crawl a webpage.

Crawl a page slowly

To add a delay to your crawler,
use -d:

python main.py -d 10 -u <url>

This will wait 10 seconds between page fetches.

Crawl only your blog

You will want to use the -i flag,
which while ignore URLs matching the passed regex::

python main.py -i "^blog" -u <url>

This will only crawl pages that contain your blog URL.