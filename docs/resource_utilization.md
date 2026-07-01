
# Resource Utilization For VectorBlox Core

Numbers measured with Libero using the PolarFire SoC Video Kit Demo. To find resource utilization metrics in Libero, view the reports tab after completing synthesis.

## Resource Utilization For VectorBlox 3.1 Core

| IP                                       | Fabric 4LUT | Fabric DFF | Interface 4LUT | Interface DFF | Total 4LUT | Total DFF | uSRAM | LSRAM | MATH |
| ---------------------------------------- | ----------- | ---------- | -------------- | ------------- | ---------- | --------- | ----- | ----- | ---- |
| VBx V1000 No Compression                 | 49694       | 51005      | 22788          | 22788         | 72482      | 73793     | 534   | 151   | 304  |
| VBx V1000 With Compression               | 62521       | 61691      | 24228          | 24228         | 86749      | 85919     | 534   | 191   | 304  |
| VBx V1000 With Unstructured Compression  | 38482       | 59332      | 34740          | 34740         | 73222      | 94072     | 450   | 526   | 289  | 
| VBx V500 No Compression                  | 30998       | 30430      | 12036          | 12036         | 43034      | 42466     | 295   | 84    | 152  |
| VBx V500 With Compression                | 36959       | 36144      | 12900          | 12900         | 49859      | 49044     | 295   | 108   | 152  |
| VBx V250 No Compression                  | 22493       | 21400      | 7260           | 7260          | 29753      | 28660     | 185   | 56    | 84   |
| VBx V250 With Compression                | 26499       | 24836      | 8016           | 8016          | 34515      | 32852     | 185   | 77    | 84   |  



## Resource Utilization For VectorBlox 3.0 Core

| IP                                        | Fabric 4LUT | Fabric DFF | Interface 4LUT | Interface DFF | Total 4LUT | Total DFF | uSRAM | LSRAM | MATH |
|-------------------------------------------|-------------|------------|----------------|---------------|------------|-----------|-------|-------|------|
| VBx V1000 No Compression                  | 47060       | 49136      | 22956          | 22956         | 70016      | 72092     | 548   | 151   | 304  |
| VBx V1000 With Compression                | 60033       | 59645      | 24360          | 24360         | 84393      | 84005     | 548   | 190   | 304  |
| VBx V1000 With  Unstructured Compression  | 38219       | 59389      | 34740          | 34740         | 72959      | 94129     | 450   | 526   | 289  |
| VBx V500 No Compression                   | 29711       | 29493      | 12204          | 12204         | 41915      | 41697     | 309   | 84    | 152  |
| VBx V500 With Compression                 | 36145       | 34951      | 13032          | 13032         | 49177      | 47983     | 309   | 107   | 152  |
| VBx V250 No Compression                   | 21550       | 20268      | 7260           | 7260          | 28810      | 27528     | 185   | 56    | 84   |
| VBx V250 With Compression                 | 25790       | 23738      | 7980           | 7980          | 33770      | 31718     | 185   | 76    | 84   |
