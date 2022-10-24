## Augment for OCR data

Function can be used for augment
| Symbol | Function
| ------------- | :-----------:
| color | Change text color
| underline | Draw underline in text
| delete | Draw delete line in text
|blur | Blur image
|stretch | Random stretch image
|distort | Random distort image
|noise | Gaussian noise
|bg | Draw ink in random position

Argument

```
python main.py --method your function --prob your prob
Example: python main.py --method blur noise --prob 0.5 0.5
Note: --method all will use all function
      --prob all will set prob of all function to 1
```
