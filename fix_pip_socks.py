"""
Fix pip SOCKS error by patching requests library
"""
import sys
import os

def patch_requests_library():
    """Patch requests library to skip SOCKS validation"""
    try:
        # Find pip's requests module
        import pip._vendor.requests.adapters as adapters_module
        import pip._vendor.urllib3.poolmanager as poolmanager_module
        
        # Backup the original methods
        original_proxy_manager_for = adapters_module.HTTPAdapter.proxy_manager_for
        original_get_connection = adapters_module.HTTPAdapter.get_connection
        
        # Create wrapper that skips SOCKS check
        def patched_proxy_manager_for(self, proxy):
            """Override to skip SOCKS validation"""
            if proxy is None:
                return None
            
            # Check for SOCKS proxy - just return None to skip it
            if proxy.lower().startswith('socks'):
                print(f"[INFO] Ignoring SOCKS proxy: {proxy}")
                return None
            
            # For other proxies, use original method
            try:
                return original_proxy_manager_for(self, proxy)
            except Exception as e:
                print(f"[WARNING] Proxy error ({proxy}): {e}")
                return None
        
        # Also patch get_connection to handle None proxy manager gracefully
        def patched_get_connection(self, url, proxies=None):
            """Override to handle None proxy managers"""
            try:
                return original_get_connection(self, url, proxies)
            except (AttributeError, TypeError) as e:
                if "NoneType" in str(e):
                    print(f"[INFO] Skipping proxy for {url}")
                    # Return direct connection without proxy
                    return original_get_connection(self, url, None)
                raise
        
        # Replace the methods
        adapters_module.HTTPAdapter.proxy_manager_for = patched_proxy_manager_for
        adapters_module.HTTPAdapter.get_connection = patched_get_connection
        print("[OK] Patched requests library successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to patch requests: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if patch_requests_library():
        # Now run pip with install command
        import subprocess
        import pip
        
        args = sys.argv[1:]  # Get all arguments after script name
        print(f"[INFO] Running: pip {' '.join(args)}")
        sys.exit(pip.main(args))
    else:
        sys.exit(1)
