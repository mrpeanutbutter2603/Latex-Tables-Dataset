To mine arxiv papers between a particular time frame this is the command:
python arxiv_mine.py --date_start 20230101000000 --date_end 20231231235959 --max_results 100
where:
--date_start: Start date in the format YYYYMMDDHHMMSS
--date_end: End date in the format YYYYMMDDHHMMSS
--max_results: Maximum number of results to return

Once, the papers are mined they would be in the directory _arxiv_sources_
Then to run table extraction:
We just need to run the command:
python te_v2.py
