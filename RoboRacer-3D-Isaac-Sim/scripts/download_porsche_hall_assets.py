#!/usr/bin/env python3
"""Download the pinned openly licensed Porsche and CC0 hall models."""
import hashlib
import json
from pathlib import Path
import uuid
import requests

ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'data/local/objects/sources';folder.mkdir(parents=True,exist_ok=True)
    for asset in json.loads((ROOT/'configs/porsche_hall_assets.json').read_text()):
        target=folder/(asset['name']+'.glb')
        if not target.exists():
            url=asset.get('download_url')
            if not url:
                response=requests.get(asset['download_api'],params={'scene_uuid':str(uuid.uuid4())},timeout=30)
                response.raise_for_status();url=response.json()['filePath']
            response=requests.get(url,headers={'User-Agent':'BlenderKit/3.16','Referer':'https://www.blenderkit.com/'},timeout=120)
            response.raise_for_status();payload=response.content
            if hashlib.sha256(payload).hexdigest()!=asset['sha256']:
                raise ValueError('Upstream asset changed; review licence/model before updating manifest: '+asset['name'])
            target.write_bytes(payload)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=asset['sha256']:
            raise ValueError('Existing asset does not match manifest: '+str(target))
        (folder/(asset['name']+'_metadata.json')).write_text(json.dumps(asset,indent=2))
        print('Verified',asset['name'],asset['license'])
    # Original Poly Haven material: the BlenderKit GLB's baked diffuse is black.
    original=folder/'plastic_crate_original'
    for item in json.loads((ROOT/'configs/plastic_crate_original_files.json').read_text()):
        target=original/item['path']
        if not target.exists():
            response=requests.get(item['url'],headers={'User-Agent':'RoboRacer-Demo/1.0'},timeout=120)
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest()!=item['sha256']:
                raise ValueError('Upstream crate file changed: '+item['path'])
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(response.content)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Crate file hash mismatch: '+item['path'])
        print('Verified original CC0 crate file',item['path'])

if __name__=='__main__':main()
