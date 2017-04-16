# [What's Cooking](https://www.kaggle.com/c/whats-cooking/data)

_Picture yourself strolling through your local, open-air market... What do you see? What do you smell? What will you make for dinner tonight?_

If you're in Northern California, you'll be walking past the inevitable bushels of leafy greens, spiked with dark purple kale and the bright pinks and yellows of chard. Across the world in South Korea, mounds of bright red kimchi greet you, while the smell of the sea draws your attention to squids squirming nearby. India’s market is perhaps the most colorful, awash in the rich hues and aromas of dozens of spices: turmeric, star anise, poppy seeds, and garam masala as far as the eye can see.

Some of our strongest geographic and cultural associations are tied to a region's local foods. This playground competitions asks you to predict the category of a dish's cuisine given a list of its ingredients. 

## Acknowledgements

We want to thank Yummly for providing this unique dataset. Kaggle is hosting this playground competition for fun and practice.

# Submission

Submissions are evaluated on the categorization accuracy (the percent of dishes that you correctly classify).

## Submission File

Your submission file should predict the cuisine for each recipe in the test set. The file should contain a header and have the following format:

```
id,cuisine
35203,italian
17600,italian
35200,italian
17602,italian
...
etc.
```

# Data

In the dataset, we include the recipe id, the type of cuisine, and the list of ingredients of each recipe (of variable length). The data is stored in JSON format. 

An example of a recipe node in train.json:

```
{
 "id": 24717,
 "cuisine": "indian",
 "ingredients": [
     "tumeric",
     "vegetable stock",
     "tomatoes",
     "garam masala",
     "naan",
     "red lentils",
     "red chili peppers",
     "onions",
     "spinach",
     "sweet potatoes"
 ]
}
```

In the test file test.json, the format of a recipe is the same as train.json, only the cuisine type is removed, as it is the target variable you are going to predict.

## File descriptions

- train.json - the training set containing recipes id, type of cuisine, and list of ingredients
- test.json - the test set containing recipes id, and list of ingredients
- sample_submission.csv - a sample submission file in the correct format
