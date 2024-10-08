from _typeshed import Incomplete

from influxdb_client.service._base_service import _BaseService

class ChecksService(_BaseService):
    def __init__(self, api_client: Incomplete | None = None) -> None: ...
    def create_check(self, post_check, **kwargs): ...
    def create_check_with_http_info(self, post_check, **kwargs): ...
    async def create_check_async(self, post_check, **kwargs): ...
    def delete_checks_id(self, check_id, **kwargs): ...
    def delete_checks_id_with_http_info(self, check_id, **kwargs): ...
    async def delete_checks_id_async(self, check_id, **kwargs): ...
    def delete_checks_id_labels_id(self, check_id, label_id, **kwargs): ...
    def delete_checks_id_labels_id_with_http_info(self, check_id, label_id, **kwargs): ...
    async def delete_checks_id_labels_id_async(self, check_id, label_id, **kwargs): ...
    def get_checks(self, org_id, **kwargs): ...
    def get_checks_with_http_info(self, org_id, **kwargs): ...
    async def get_checks_async(self, org_id, **kwargs): ...
    def get_checks_id(self, check_id, **kwargs): ...
    def get_checks_id_with_http_info(self, check_id, **kwargs): ...
    async def get_checks_id_async(self, check_id, **kwargs): ...
    def get_checks_id_labels(self, check_id, **kwargs): ...
    def get_checks_id_labels_with_http_info(self, check_id, **kwargs): ...
    async def get_checks_id_labels_async(self, check_id, **kwargs): ...
    def get_checks_id_query(self, check_id, **kwargs): ...
    def get_checks_id_query_with_http_info(self, check_id, **kwargs): ...
    async def get_checks_id_query_async(self, check_id, **kwargs): ...
    def patch_checks_id(self, check_id, check_patch, **kwargs): ...
    def patch_checks_id_with_http_info(self, check_id, check_patch, **kwargs): ...
    async def patch_checks_id_async(self, check_id, check_patch, **kwargs): ...
    def post_checks_id_labels(self, check_id, label_mapping, **kwargs): ...
    def post_checks_id_labels_with_http_info(self, check_id, label_mapping, **kwargs): ...
    async def post_checks_id_labels_async(self, check_id, label_mapping, **kwargs): ...
    def put_checks_id(self, check_id, check, **kwargs): ...
    def put_checks_id_with_http_info(self, check_id, check, **kwargs): ...
    async def put_checks_id_async(self, check_id, check, **kwargs): ...
