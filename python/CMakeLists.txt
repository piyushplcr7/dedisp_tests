install(
    FILES
        filterbank.py
        plot_burst.py
        psr_constants.py
        sigproc.py
    DESTINATION bin/python
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_READ  GROUP_EXECUTE
        WORLD_READ WORLD_READ  WORLD_EXECUTE
)
