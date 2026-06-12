# Industrial Multimodal Sample Sources

These files are small teaching fixtures derived from public web material and local synthetic plant data. They are intentionally compact so ingestion and eval can run quickly.

## Public Web References

- OSHA, Control of Hazardous Energy (Lockout/Tagout) program page: https://www.osha.gov/control-hazardous-energy/program
- OSHA, 29 CFR 1910.147 control of hazardous energy standard: https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- U.S. Department of Energy, Improving Compressed Air System Performance: A Sourcebook for Industry: https://www.energy.gov/sites/default/files/2016/03/f30/Improving%20Compressed%20Air%20Sourcebook%20version%203.pdf

## Fixture Map

- `files/loto_checklist.md`: Markdown maintenance checklist adapted as a Chinese teaching note from OSHA LOTO concepts.
- `files/energy_control.html`: HTML procedure page for hazardous energy isolation and shift transfer.
- `files/compressed_air_excerpt.txt`: Plain text compressed-air troubleshooting excerpt based on DOE sourcebook topics.
- `files/compressed_air_sourcebook_sample.pdf`: Tiny PDF fixture that preserves PDF page metadata for compressed-air leak and pressure-drop retrieval.
- `tables/pump_vibration_readings.csv`: CSV sensor rows for pump vibration alarms and bearing inspection.
- `tables/compressor_energy.tsv`: TSV energy rows for compressed-air demand and leak-rate examples.
- `images/compressed_air_dashboard.png`: Local teaching diagram derived from compressed-air system sourcebook concepts.
- `sample_images_industrial.jsonl`: OCR/caption metadata for the image fixture.

The fixture wording is not a verbatim copy of the source pages. It is shaped for RAG lessons: each document contains explicit operational facts, source metadata, and query terms that should be recoverable through source filters.
