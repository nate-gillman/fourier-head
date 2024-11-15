# website-demo

The project page contains a javascript visualization of the Fourier head learning a few different densities.
We include the details for generating those as well.
To generate the json files used in the website, run the following script, which will output the jsons into the `output` directory.

```bash
conda activate chronos
python run_exp.py
```

We also include a script which generates visualizations of the learned Fourier PDFs, along with the discretized versions of them.
Running the following will write the graphs into the `misc/website-demo/output/discretization_graphs` directory.

```bash
python misc/website-demo/graph_discretization_procedure.py
```