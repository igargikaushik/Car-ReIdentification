Color feature is used to filter out some data, but try not to leak

Because of the angle problem of the captured image, mirror reversal may be performed when matching image features.

The traditional method uses three characteristics: color, model, rear window (person matching, person very similar to a car, otherwise not used characteristics, person here only increases confidence)

2024 09 17  test -- 1000 pictures contain one correct one, final test top5 correctness rate 26%

First test, check whether the color filter is leaking

Traditional method (filtering only excludes bad data, no need for top5)

When testing car window, you can find some good data test in advance

color filter
model selection
Car Window Filtering (only matching car window left area only)
deep learning method

vehicle detection(network)
vehicle comparison
front windshield glass recognition
human face comparison

2024 09 18test--conducted the first color filter on the photo, found that 98% of the filters were successful, and the remaining 20%

A picture below

Query Image： Background interference, overall judgment yellow and orange
