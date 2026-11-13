## rt_civitai and rt_net tests, no network
import io, os, sys, copy, unittest, urllib.error
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rt_net, rt_civitai

VID_URL = "https://image.civitai.com/abc/uuid1/width=450/111.mp4"
IMG_URL = "https://image.civitai.com/abc/uuid2/width=450/222.jpeg"
VERSION = {"id": 456, "modelId": 123, "name": "v2.0", "baseModel": "Illustrious",
           "trainedWords": ["glowing eyes", " neon ", "", "glowing eyes"],
           "model": {"name": "Neon Dreams", "type": "LORA"},
           "images": [{"url": VID_URL, "type": "video"}, {"url": IMG_URL, "type": "image"}],
           "files": [{"name": "neon_vae.pt", "primary": False, "downloadUrl": "https://civitai.com/api/download/models/456?type=VAE"},
                     {"name": "neonDreams_v20.safetensors", "primary": True, "hashes": {"SHA256": "ABCDEF"},
                      "downloadUrl": "https://civitai.com/api/download/models/456"}]}

def http_error(code):
    return urllib.error.HTTPError("https://civitai.com/x", code, "err", {}, io.BytesIO())

class FakeGet:
    def __init__(self, routes, reject_key=False):
        self.routes, self.reject_key, self.calls = routes, reject_key, []

    def __call__(self, url, api_key=None, timeout=15, retries=3):
        self.calls.append((url, api_key))
        if self.reject_key and api_key:
            raise http_error(401)
        for frag, payload in self.routes.items(): ## first matching route wins
            if frag in url:
                return copy.deepcopy(payload)
        raise http_error(404)

class CivitaiTests(unittest.TestCase):
    def setUp(self):
        self.real_get, self.logged = rt_net.http_get_json, []
        rt_civitai.key_rejected, rt_civitai.LOG = False, self.logged.append

    def tearDown(self):
        rt_net.http_get_json, rt_civitai.LOG, rt_civitai.key_rejected = self.real_get, print, False

    def test_parse_url(self):
        cases = {"https://civitai.com/models/123/neon-dreams?modelVersionId=456": ("version", 456),
                 "https://civitai.com/api/download/models/789": ("version", 789),
                 "https://civitai.com/api/v1/model-versions/42": ("version", 42),
                 "https://civitai.com/models/123/neon-dreams": ("model", 123),
                 "https://huggingface.co/models/123": (None, None),
                 "https://civitai.com/user/bob": (None, None), "": (None, None)}
        for url, want in cases.items():
            self.assertEqual(rt_civitai.parse_url(url), want, url)

    def test_is_civitai(self):
        self.assertTrue(rt_civitai.is_civitai("https://civitai.com/models/1"))
        self.assertFalse(rt_civitai.is_civitai("https://notcivitai.com/models/1"))

    def test_tag_for_base_model(self):
        self.assertEqual(rt_civitai.tag_for_base_model("SDXL 1.0"), "[SDXL 1.0]")
        self.assertEqual(rt_civitai.tag_for_base_model("Turbo 9"), "[Turbo 9]")
        self.assertIsNone(rt_civitai.tag_for_base_model(""))

    def test_preview_candidates_video_first(self):
        got = rt_civitai.preview_candidates([{"url": VID_URL, "type": "video"}, {"url": IMG_URL, "type": "image"}, {"url": IMG_URL}])
        self.assertEqual(got, [IMG_URL, "https://image.civitai.com/abc/uuid1/anim=false,width=450/111.mp4",
                               "https://image.civitai.com/abc/uuid1/width=450/111.jpeg"])
        self.assertEqual(rt_civitai.preview_candidates([]), [])

    def test_normalize_fixture(self):
        rec = rt_civitai.normalize(VERSION, source="hash")
        self.assertEqual((rec["model_id"], rec["version_id"], rec["model_name"], rec["version_name"]), (123, 456, "Neon Dreams", "v2.0"))
        self.assertEqual((rec["base_model"], rec["type"], rec["source"]), ("Illustrious", "LORA", "hash"))
        self.assertEqual(rec["tags"], ["glowing eyes", "neon"])
        self.assertEqual([i["type"] for i in rec["images"]], ["video", "image"])
        self.assertEqual(rec["download_url"], "https://civitai.com/api/download/models/456")
        self.assertEqual((rec["file_name"], rec["sha256"]), ("neonDreams_v20.safetensors", "abcdef"))
        self.assertEqual(rec["raw"]["trainedWords"], VERSION["trainedWords"])

    def test_normalize_fills_model_name(self):
        ver = {k: v for k, v in VERSION.items() if k not in ("model", "modelId")}
        rec = rt_civitai.normalize(ver, {"id": 123, "name": "Neon Dreams", "type": "LORA"})
        self.assertEqual((rec["model_id"], rec["model_name"], rec["raw"]["model"]["name"]), (123, "Neon Dreams", "Neon Dreams"))
        self.assertIsNone(rt_civitai.normalize(None)["model_id"])

    def test_fetch_by_url_and_hash(self):
        rt_net.http_get_json = FakeGet({"/model-versions/by-hash/": VERSION, "/model-versions/456": VERSION})
        rec = rt_civitai.fetch_by_url("https://civitai.com/models/123?modelVersionId=456", "KEY")
        self.assertEqual((rec["version_id"], rec["tags"]), (456, ["glowing eyes", "neon"]))
        self.assertEqual(rt_civitai.fetch_by_hash("abcdef", "KEY")["source"], "hash")
        rt_net.http_get_json = FakeGet({})
        self.assertIsNone(rt_civitai.fetch_by_hash("abcdef", "KEY"))
        self.assertIn("404", rt_civitai.last_error)

    def test_fetch_model_first_version(self):
        ver = {k: v for k, v in VERSION.items() if k not in ("model", "modelId")}
        rt_net.http_get_json = FakeGet({"/models/123": {"id": 123, "name": "Neon Dreams", "type": "LORA", "modelVersions": [ver]}})
        rec = rt_civitai.fetch_model(123, None)
        self.assertEqual((rec["model_name"], rec["version_id"]), ("Neon Dreams", 456))

    def test_images_fallback(self):
        bare = dict(VERSION, images=[])
        fake = FakeGet({"/images?modelVersionId=456": {"items": [{"url": IMG_URL, "type": "image"}]}, "/model-versions/456": bare})
        rt_net.http_get_json = fake
        rec = rt_civitai.fetch_version(456, "KEY")
        self.assertEqual(rec["images"], [{"url": IMG_URL, "type": "image"}])
        self.assertEqual(rec["raw"]["images"], rec["images"])
        self.assertTrue(all(k == "KEY" for _, k in fake.calls))
        rt_net.http_get_json = FakeGet({"/images?": {"items": []}, "/models/123": {"modelVersions": [{"id": 456, "images": [{"url": VID_URL}]}]}})
        self.assertEqual(rt_civitai.fetch_images_fallback({"version_id": 456, "model_id": 123}, None)[0]["type"], "video")

    def test_key_rejected_once(self):
        fake = FakeGet({"/model-versions/456": VERSION}, reject_key=True)
        rt_net.http_get_json = fake
        self.assertEqual(rt_civitai.fetch_version(456, "BAD")["version_id"], 456)
        self.assertEqual(rt_civitai.fetch_version(456, "BAD")["version_id"], 456)
        self.assertEqual(len(self.logged), 1)
        self.assertIn("API key rejected", self.logged[0])
        self.assertEqual([k for _, k in fake.calls], ["BAD", None, None])

    def test_network_failure_returns_none(self):
        def boom(*args, **kwargs):
            raise urllib.error.URLError("offline")
        rt_net.http_get_json = boom
        self.assertIsNone(rt_civitai.fetch_by_url("https://civitai.com/models/123", "KEY"))
        self.assertIsNone(rt_civitai.fetch_by_url("https://example.com/file.safetensors", "KEY"))

class NetTests(unittest.TestCase):
    def test_redirect_header_strip(self):
        handler = rt_net.RedirectHandler()
        req = rt_net._request("https://civitai.com/api/download/models/456", "KEY")
        self.assertTrue(req.has_header("Authorization"))
        cdn = handler.redirect_request(req, None, 302, "Found", {}, "https://b2.example-cdn.com/f.safetensors?sig=x")
        self.assertFalse(cdn.has_header("Authorization"))
        same = handler.redirect_request(req, None, 302, "Found", {}, "https://civitai.com/api/download/models/456?x=1")
        self.assertEqual(same.get_header("Authorization"), "Bearer KEY")

    def test_key_never_sent_off_civitai(self):
        self.assertFalse(rt_net._request("https://huggingface.co/x", "KEY").has_header("Authorization"))
        self.assertTrue(rt_net._request("https://image.civitai.com/x.jpeg", "KEY").has_header("Authorization"))
        self.assertIn("Mozilla", rt_net._request("https://civitai.com/x").get_header("User-agent"))

    def test_image_magic(self):
        self.assertTrue(rt_net.looks_like_image(bytes([0xFF, 0xD8, 0xFF, 0xE0])))
        self.assertTrue(rt_net.looks_like_image(b"RIFF0000WEBPVP8 "))
        self.assertFalse(rt_net.looks_like_image(b"<!DOCTYPE html>"))

if __name__ == "__main__":
    unittest.main()
