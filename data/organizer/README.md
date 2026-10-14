# Organizer dataset drop-in

Place the organizers' files here. The loader picks them up automatically and labels runs ORGANIZER.

Required tables (csv, json or xlsx): batches, destinations, lanes, prices, demand.
Optional: hubs, commodities, cost_model.yaml (overrides config/cost_model.yaml), scenarios/*.yaml.

If file or column names differ, copy mapping.example.yaml to mapping.yaml and edit it.
Run 'python -m freshroute.adapters.check' to validate the files and get readable errors.
