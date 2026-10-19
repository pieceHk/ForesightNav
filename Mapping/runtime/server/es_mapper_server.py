"""Compatibility imports for the native PBSP mapping worker."""

from runtime.mapping_worker import MappingServer as ESMapperServer, main
from runtime.protocol import decode_binary_payload as _decode_binary_payload
from runtime.protocol import handle_connection as _handle_conn, serve


if __name__ == "__main__":
    main()
