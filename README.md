# Adjacent Block Indexing for Record Linkage

The code in this repository is intended for use with the [recordlinkage](http://recordlinkage.readthedocs.io/en/latest/)
package.  It contains an implementation of the Adjacent Block Indexing method.  
This method combines some features of Standard Blocking, Progressive
Blocking and Sorted Neighbourhood Indexing.  In addition, it also allows for
meaningful treatment of missing values and the simultaneous use of multiple
sorting orders for the blocks (ie: sorting by each of the blocking keys
individually).  

IPython Notebooks in this repo perform numerical experiments in which Adjacent Block
Indexing is compared to Standard Blocking and Sorted Neighbourhood Indexing.  These
take some time to run.  Their results are that *under the conditions tested*:

* Adjacent Block
Indexing has similar scalability properties to Sorted Neighbourhood Indexing, Standard Blocking
and Full Indexing (ie: runtime is approximately linear with respect to the size of the 
index produced), and
* Compared to the other methods, Adjacent Block Indexing produces superior index 
quality at the expense of increased runtime.


