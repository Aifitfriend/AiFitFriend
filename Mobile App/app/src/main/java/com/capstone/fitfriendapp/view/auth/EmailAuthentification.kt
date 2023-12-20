package com.capstone.fitfriendapp.view.auth

import android.content.Context
import android.text.Editable
import android.text.TextWatcher
import android.util.AttributeSet
import android.util.Patterns
import com.google.android.material.textfield.TextInputEditText

class EmailAuthentification : TextInputEditText {
    constructor(context: Context) : super(context){
        init()
    }
    constructor(context: Context, attrs : AttributeSet) : super(context, attrs){
        init()
    }
    constructor(context: Context, attrs: AttributeSet, defStyleAttr: Int) : super(context, attrs, defStyleAttr){
        init()
    }

    private fun init() {
        addTextChangedListener(object : TextWatcher{
            override fun beforeTextChanged(s: CharSequence, start: Int, after: Int, count: Int) {

            }

            override fun onTextChanged(s: CharSequence, start: Int, before: Int, count: Int) {
                if (!s.isNullOrEmpty() && !Patterns.EMAIL_ADDRESS.matcher(s).matches()){
                    setError("Email tidak Valid")
                }else{
                    error = null
                }
            }

            override fun afterTextChanged(s: Editable?) {

            }
        })
    }
}