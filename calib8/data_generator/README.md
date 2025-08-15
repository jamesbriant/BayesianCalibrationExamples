# Data Generating Process

## Generating Data

### Step 1: `cd` into `calib8/`

```bash
cd calib8/
```

### Step 2: Check/Edit the config file

Check the config files used for generating the datasets.

### Step 3: Run the Generate Script

From within the `calib8/` directory run:

```bash
python -m data_generator.generate --config <FILE_NAME>
```

Note: Do not include the file extension in the `<FILE_NAME>`. eg. `calib8`.

## Verify Data

Check the data is in the correct format and can be loaded into a `kohgpjax.KOHDataset` object.

```bash
python -m verify_data --file-name calib8
```
