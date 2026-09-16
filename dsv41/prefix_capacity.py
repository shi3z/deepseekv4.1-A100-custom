"""Capacity calculator for DeepSeek V4.1 prefix caches.

Uses inference/config.json and the runtime cache topology (max_seqs=1,
full mirror count) rather than hard-coded dimensions.
"""
import argparse, json, os

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='/mnt/ssdraid/models/DeepSeek-V4.1-Flash-Abliterated/inference/config.json'); ap.add_argument('--devices',type=int,default=5); ap.add_argument('--anchors',default='1,2,4,16,100'); args=ap.parse_args()
    c=json.load(open(args.config)); ratios=c['compress_ratios']; owners=[o for o in c['kv_source_layers'] if o < c['n_layers']]
    hd,ik=c['head_dim'],c['index_head_dim']; bpe=2
    print(f"head_dim={hd} index_head_dim={ik} dtype=bf16 bytes/row={(hd+ik)*bpe} mirrors={args.devices}")
    print('owner ratio rows@1M canonical_mib mirrored_mib')
    canon_1m=0
    for o in owners:
      r=ratios[o]; rows=1048576//r+1; mib=rows*(hd+ik)*bpe/2**20; canon_1m+=mib
      print(f'{o:5d} {r:5d} {rows:10d} {mib:16.1f} {mib*args.devices:14.1f}')
    print(f'canonical@1M={canon_1m:.1f}MiB mirrored@1M={canon_1m*args.devices:.1f}MiB')
    anchors=[1,2,4,16,100]
    print('tokens canonical_mib mirrored_mib ' + ' '.join(f'anchors{a}_mib' for a in anchors))
    for tok in (75000,131072,262144,524288,1048576):
      canon=sum((tok//ratios[o]+1)*(hd+ik)*bpe for o in owners)/2**20
      mir=canon*args.devices
      print(f"{tok:7d} {canon:14.1f} {mir:12.1f} " + ' '.join(f'{mir*a:14.1f}' for a in anchors))
    print('anchors:',args.anchors)
if __name__=='__main__': main()
