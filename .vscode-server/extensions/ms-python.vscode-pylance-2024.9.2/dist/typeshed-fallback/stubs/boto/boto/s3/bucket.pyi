from _typeshed import Incomplete
from typing import Any

from .bucketlistresultset import BucketListResultSet
from .connection import S3Connection
from .key import Key

class S3WebsiteEndpointTranslate:
    trans_region: dict[str, str]
    @classmethod
    def translate_region(cls, reg: str) -> str: ...

S3Permissions: list[str]

class Bucket:
    LoggingGroup: str
    BucketPaymentBody: str
    VersioningBody: str
    VersionRE: str
    MFADeleteRE: str
    name: str
    connection: S3Connection
    key_class: type[Key]
    def __init__(self, connection: S3Connection | None = None, name: str | None = None, key_class: type[Key] = ...) -> None: ...
    def __iter__(self): ...
    def __contains__(self, key_name) -> bool: ...
    def startElement(self, name, attrs, connection): ...
    creation_date: Any
    def endElement(self, name, value, connection): ...
    def set_key_class(self, key_class): ...
    def lookup(self, key_name, headers: dict[str, str] | None = None): ...
    def get_key(
        self,
        key_name,
        headers: dict[str, str] | None = None,
        version_id: Incomplete | None = None,
        response_headers: dict[str, str] | None = None,
        validate: bool = True,
    ) -> Key: ...
    def list(
        self,
        prefix: str = "",
        delimiter: str = "",
        marker: str = "",
        headers: dict[str, str] | None = None,
        encoding_type: Incomplete | None = None,
    ) -> BucketListResultSet: ...
    def list_versions(
        self,
        prefix: str = "",
        delimiter: str = "",
        key_marker: str = "",
        version_id_marker: str = "",
        headers: dict[str, str] | None = None,
        encoding_type: str | None = None,
    ) -> BucketListResultSet: ...
    def list_multipart_uploads(
        self,
        key_marker: str = "",
        upload_id_marker: str = "",
        headers: dict[str, str] | None = None,
        encoding_type: Incomplete | None = None,
    ): ...
    def validate_kwarg_names(self, kwargs, names): ...
    def get_all_keys(self, headers: dict[str, str] | None = None, **params): ...
    def get_all_versions(self, headers: dict[str, str] | None = None, **params): ...
    def validate_get_all_versions_params(self, params): ...
    def get_all_multipart_uploads(self, headers: dict[str, str] | None = None, **params): ...
    def new_key(self, key_name: Incomplete | None = None): ...
    def generate_url(
        self,
        expires_in,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        force_http: bool = False,
        response_headers: dict[str, str] | None = None,
        expires_in_absolute: bool = False,
    ): ...
    def delete_keys(
        self, keys, quiet: bool = False, mfa_token: Incomplete | None = None, headers: dict[str, str] | None = None
    ): ...
    def delete_key(
        self,
        key_name,
        headers: dict[str, str] | None = None,
        version_id: Incomplete | None = None,
        mfa_token: Incomplete | None = None,
    ): ...
    def copy_key(
        self,
        new_key_name,
        src_bucket_name,
        src_key_name,
        metadata: Incomplete | None = None,
        src_version_id: Incomplete | None = None,
        storage_class: str = "STANDARD",
        preserve_acl: bool = False,
        encrypt_key: bool = False,
        headers: dict[str, str] | None = None,
        query_args: Incomplete | None = None,
    ): ...
    def set_canned_acl(
        self, acl_str, key_name: str = "", headers: dict[str, str] | None = None, version_id: Incomplete | None = None
    ): ...
    def get_xml_acl(self, key_name: str = "", headers: dict[str, str] | None = None, version_id: Incomplete | None = None): ...
    def set_xml_acl(
        self,
        acl_str,
        key_name: str = "",
        headers: dict[str, str] | None = None,
        version_id: Incomplete | None = None,
        query_args: str = "acl",
    ): ...
    def set_acl(
        self, acl_or_str, key_name: str = "", headers: dict[str, str] | None = None, version_id: Incomplete | None = None
    ): ...
    def get_acl(self, key_name: str = "", headers: dict[str, str] | None = None, version_id: Incomplete | None = None): ...
    def set_subresource(
        self, subresource, value, key_name: str = "", headers: dict[str, str] | None = None, version_id: Incomplete | None = None
    ): ...
    def get_subresource(
        self, subresource, key_name: str = "", headers: dict[str, str] | None = None, version_id: Incomplete | None = None
    ): ...
    def make_public(self, recursive: bool = False, headers: dict[str, str] | None = None): ...
    def add_email_grant(self, permission, email_address, recursive: bool = False, headers: dict[str, str] | None = None): ...
    def add_user_grant(
        self,
        permission,
        user_id,
        recursive: bool = False,
        headers: dict[str, str] | None = None,
        display_name: Incomplete | None = None,
    ): ...
    def list_grants(self, headers: dict[str, str] | None = None): ...
    def get_location(self): ...
    def set_xml_logging(self, logging_str, headers: dict[str, str] | None = None): ...
    def enable_logging(
        self, target_bucket, target_prefix: str = "", grants: Incomplete | None = None, headers: dict[str, str] | None = None
    ): ...
    def disable_logging(self, headers: dict[str, str] | None = None): ...
    def get_logging_status(self, headers: dict[str, str] | None = None): ...
    def set_as_logging_target(self, headers: dict[str, str] | None = None): ...
    def get_request_payment(self, headers: dict[str, str] | None = None): ...
    def set_request_payment(self, payer: str = "BucketOwner", headers: dict[str, str] | None = None): ...
    def configure_versioning(
        self, versioning, mfa_delete: bool = False, mfa_token: Incomplete | None = None, headers: dict[str, str] | None = None
    ): ...
    def get_versioning_status(self, headers: dict[str, str] | None = None): ...
    def configure_lifecycle(self, lifecycle_config, headers: dict[str, str] | None = None): ...
    def get_lifecycle_config(self, headers: dict[str, str] | None = None): ...
    def delete_lifecycle_configuration(self, headers: dict[str, str] | None = None): ...
    def configure_website(
        self,
        suffix: Incomplete | None = None,
        error_key: Incomplete | None = None,
        redirect_all_requests_to: Incomplete | None = None,
        routing_rules: Incomplete | None = None,
        headers: dict[str, str] | None = None,
    ): ...
    def set_website_configuration(self, config, headers: dict[str, str] | None = None): ...
    def set_website_configuration_xml(self, xml, headers: dict[str, str] | None = None): ...
    def get_website_configuration(self, headers: dict[str, str] | None = None): ...
    def get_website_configuration_obj(self, headers: dict[str, str] | None = None): ...
    def get_website_configuration_with_xml(self, headers: dict[str, str] | None = None): ...
    def get_website_configuration_xml(self, headers: dict[str, str] | None = None): ...
    def delete_website_configuration(self, headers: dict[str, str] | None = None): ...
    def get_website_endpoint(self): ...
    def get_policy(self, headers: dict[str, str] | None = None): ...
    def set_policy(self, policy, headers: dict[str, str] | None = None): ...
    def delete_policy(self, headers: dict[str, str] | None = None): ...
    def set_cors_xml(self, cors_xml, headers: dict[str, str] | None = None): ...
    def set_cors(self, cors_config, headers: dict[str, str] | None = None): ...
    def get_cors_xml(self, headers: dict[str, str] | None = None): ...
    def get_cors(self, headers: dict[str, str] | None = None): ...
    def delete_cors(self, headers: dict[str, str] | None = None): ...
    def initiate_multipart_upload(
        self,
        key_name,
        headers: dict[str, str] | None = None,
        reduced_redundancy: bool = False,
        metadata: Incomplete | None = None,
        encrypt_key: bool = False,
        policy: Incomplete | None = None,
    ): ...
    def complete_multipart_upload(self, key_name, upload_id, xml_body, headers: dict[str, str] | None = None): ...
    def cancel_multipart_upload(self, key_name, upload_id, headers: dict[str, str] | None = None): ...
    def delete(self, headers: dict[str, str] | None = None): ...
    def get_tags(self): ...
    def get_xml_tags(self): ...
    def set_xml_tags(self, tag_str, headers: dict[str, str] | None = None, query_args: str = "tagging"): ...
    def set_tags(self, tags, headers: dict[str, str] | None = None): ...
    def delete_tags(self, headers: dict[str, str] | None = None): ...
