plugins { id("com.android.asset-pack") }

assetPack {
    packName.set("models_cpu")
    dynamicDelivery { deliveryType.set("on-demand") }
}
