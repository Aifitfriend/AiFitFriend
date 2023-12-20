package com.capstone.fitfriendapp.view.home

import android.os.Parcel
import android.os.Parcelable

data class WorkOut(
    val name : String?,
    val photo: Int

):Parcelable{
    constructor(parcel: Parcel) : this(
        parcel.readString(),
        parcel.readInt()
    ) {
    }

    override fun writeToParcel(parcel: Parcel, flags: Int) {
        parcel.writeString(name)

        parcel.writeInt(photo)
    }

    override fun describeContents(): Int {
        return 0
    }

    companion object CREATOR : Parcelable.Creator<WorkOut> {
        override fun createFromParcel(parcel: Parcel): WorkOut {
            return WorkOut(parcel)
        }

        override fun newArray(size: Int): Array<WorkOut?> {
            return arrayOfNulls(size)
        }
    }
}
