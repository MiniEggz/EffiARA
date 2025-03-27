Allocating and Reallocating Samples to Annotators
=================================================

One of the main features of EffiARA is the ability to 
allocate samples to a set of annotators according to user-specified
criteria. We'll here discuss in more detail how this is done within EffiARA.


Sample Allocation
-----------------

Allocating samples to annotators requires the user to input values 


 * :code:`annotation_rate` (:math:`\rho`): The estimated number of samples an annotator will complete in an hour.
 * :code:`time_available` (:math:`t`): The total number of hours available for each annotator.
 * :code:`double_proportion` (:math:`d`): The proportion of samples that will be allocated to 2 annotators, for computing inter-annotator agreement.
 * :code:`re_proportion` (:math:`r`): The proportion of samples that will be allocated to the same annotator twice, for computing intra-annotator agreement.

These variables are related to each other according to the equation below.


:math:`k = (2d + (1 + r)(1 - d))^{-1} \cdot \rho \cdot t \cdot n`


where :math:`k` is the total number of samples to annotate.

EffiARA solves this equation and then allocates samples to annotators in such a way that there is a
maximal set of points to compute inter-annotator agreement between each pair of annotators. 
While it is often the case that we want to determine the number of samples we can annotate, EffiARA
can solve for any variable in this equation given the other four. For example, we could determine
the time required to annotate a given number of samples provided an estimate of the annotation rate.


Using :code:`SampleDistributor`
...............................


Allocation is done using the :code:`SampleDistributor` class. It is initialized by providing 
four of the five variables above, with the missing variable being the one we wish to solve for.
It then solves the aforementioned equation for the fifth variable and assigns samples accordingly.


.. code-block:: python

   from effiara.preparation import SampleDistributor

   # Generate some dummy data to allocate.
   df = pd.DataFrame({"sample_id": range(1000), "value": np.random.randint(5, size=(1000, 2))})

   distrib = SampleDistributor(
       num_annotators=4,
       num_samples=None,  # We want to solve for this
       annotation_rate=20,
       time_available=4,
       double_proportion=1.0,  # double-annotate all samples
       re_proportion=0.5,      # annotators re-annotate half of their samples
   )

   distrib.set_project_distribution()  # solve the equation above
   allocations = distrib.distribute_samples(df.copy())

:code:`allocations` is a Python :code:`dict` of usernames to Pandas DataFrames indicating the samples
allocated to each annotator.


Sample Reallocation
-------------------

Occasionally, we may want to assign already annotated samples to annotators that have not seen them before.
For example, perhaps we want a third annotation for samples where the first two annotators disagreed.
This can be done using the :code:`SampleRedistributor` class. It is initialized the same as 
:code:`SampleDistributor`, but :code:`double_proportion` and :code:`re_proportion` are always set to 
0.0.

Because one of the primary uses of the :code:`SampleRedistributor` is to assign an additional annotator
to our samples, we can also initialize it from our existing :code:`SampleDistributor` instance.

.. code-block:: python

   from effiara.preparation import SampleRedistributor
   redistrib = SampleRedistributor.from_sample_distributor(distrib)
   redistrib.set_project_distribution()
   reallocations = redistrib.distribute_samples(annotated_df)


Reallocation uses a different algorithm for allocating samples which ensures that no sample is assigned to
an annotator who has already annotated it. As such, its :code:`distribute_samples` method requires
a DataFrame with annotation columns (i.e., columns in the :code:`{username}_label` format).
