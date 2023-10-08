# T-brain-SinoPac-AIGO
## Done
- Have to fix the `y contains previously unseen labels` error.

## TO-DO
- Remember if you seperate the predict funtion, when facing different unknown labels the LabelEncoding will change. -> Have to code it in the same file.(main)
- lots of external_data try to use `Longitude` and `Latitude` and calculate each different installations and put (0,1) to recognized is it nearby or not.(Some are -1, have to ignore.)
- exacute some analysis from `pairplot_output.png`
    - Building_Area: against different Cities.
    - Age_of_Building: have to times -1
    - Parking_Area: Most are 0, and if it have wont effect the result(0,1)
    - Auxiliary_Building_Area: Most are 0, and if it have wont effect the result(0,1)
