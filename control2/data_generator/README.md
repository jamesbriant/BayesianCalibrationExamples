# Data Generating Process

## Generating Data

### Step 1: `cd` into `control2/`

```bash
cd control2/
```

### Step 2: Check/Edit the config file

Check the config files used for generating the datasets.

### Step 3: Run the Generate Script

From within the `control2/` directory run:

```bash
python -m data_generator.generate --config <FILE_NAME>
```

Note: Do not include the file extension in the `<FILE_NAME>`. eg. `config_sin-a`.

## Verify Data

Check the data is in the correct format and can be loaded into a `kohgpjax.KOHDataset` object.

```bash
python -m verify_data --file-name sin-a
```
