## Use case: Capestart data
0. Fresh segmentations (e.g., by capestart) are converted to lms by 'misc' plugin in Slicer.
1. Add new cases to `/s/xnat_shadow/nodes/images and /lms` folders. 

update Datasource whenever you add new cases so they are registered in it
    test = False
    ds = Datasource(
        folder=Path("/s/xnat_shadow/nodes"), name="nodes", alias="nodes", test=test
    )
    ds.process()


now update project by (re) adding datasource
```
    P.add_data([DS.nodes, DS.nodesthick])
```

