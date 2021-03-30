install(
    PROGRAMS
        run_benchmarks.py
        run_benchmarks_analysis.py
    DESTINATION bin/python
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_READ  GROUP_EXECUTE
        WORLD_READ WORLD_READ  WORLD_EXECUTE
)