plugins { id("com.android.asset-pack") }

assetPack {
    packName.set("models_npu")
    dynamicDelivery { deliveryType.set("on-demand") }
}
