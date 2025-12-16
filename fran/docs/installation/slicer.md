
With a new slicer build, you need to find the slicerpython exe first. then navigate to code/fran/ source folder and run:        
you will need to generate constraints by using: 
uv pip compile pyproject.toml -o constraints.txt
then edit constraint.txt and remove all entries of your custom git repos (e.g., label-analysis, utilz)
`/home/ub/programs/Slicer-5.10.0-linux-amd64/bin/PythonSlicer -m pip install -c constraints.txt -e . `

