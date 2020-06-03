# Likes Prediction
We have to predict no of likes on a post.<br>
<b>Given</b> - Post Id, User Id, Category, Country, #Comments, #Views.<br>
<b>Prediction</b> - #likes

All the data has been scrapped from social media website.

## Data
There are two datasets
- train.csv​ - This is the training data set. Candidates can use this data to train & build the
model. There are 369,921 posts in this data set
- test_without_truth.csv​ - After your model is ready and optimized, you can run your model
on this test data and add the predicted likes against all the posts in this data set. The
test data has 158,542 posts
Link: https://drive.google.com/file/d/1n3ctlDQFGOzAAcLJFsuM2dwZFfCGdHgL/view?usp=sharing

## Dataset Information
- post_id​ is the unique identifier for every post
- user_id​ is the unique identifier for the publisher who published the post
- country​ is the primary geolocation of the publisher/user
- category​ is the interest that best describes the post content
- #views​ is the number of times the post is viewed by the content consumers
- #comments​ is the number of times the content consumers commented on the post
- #likes​ is the number of times the content consumers liked the post. This is the variable
that the model needs to predict
