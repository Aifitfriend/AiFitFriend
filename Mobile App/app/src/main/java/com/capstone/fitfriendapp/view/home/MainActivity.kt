package com.capstone.fitfriendapp.view.home

import android.content.Intent
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.RelativeLayout
import androidx.activity.viewModels
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.capstone.fitfriendapp.R
import com.capstone.fitfriendapp.data.pref.UserModel
import com.capstone.fitfriendapp.databinding.ActivityMainBinding
import com.capstone.fitfriendapp.view.ViewModelFactory
import com.capstone.fitfriendapp.view.getstarted.GetStartedActivity
import com.capstone.fitfriendapp.view.profile.ProfileActivity
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var rvWorkOut: RecyclerView
    private val list = ArrayList<WorkOut>()

    private val viewModel: MainViewModel by viewModels {
        ViewModelFactory.getInstance(this)
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val profileSummary: RelativeLayout = findViewById(R.id.profile_summary)
        profileSummary.setOnClickListener {
            val moveIntent = Intent(this@MainActivity, ProfileActivity::class.java)
            startActivity(moveIntent)
        }

        rvWorkOut = findViewById(R.id.rv_homePage)
        rvWorkOut.setHasFixedSize(true)

        list.addAll(getListWorkOut())
        showRecyclerList()

        observeSession()
        setupView()
    }

    private fun  getListWorkOut(): ArrayList<WorkOut>{
        val dataNama = resources.getStringArray(R.array.data_name)
        val dataPhoto = resources.obtainTypedArray(R.array.data_photo)

        val listWorkOut = ArrayList<WorkOut>()
        for (i in dataNama.indices){
            val workOut = WorkOut(dataNama[i],dataPhoto.getResourceId(i,-1))
            listWorkOut.add(workOut)
        }
        return listWorkOut
    }

    private fun showRecyclerList(){
        rvWorkOut.layoutManager = LinearLayoutManager(this)
        val listWorkOutAdapter = ListWorkOutAdapter(list)
        rvWorkOut.adapter=listWorkOutAdapter
    }


    private fun setupView() {
        @Suppress("DEPRECATION")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.insetsController?.hide(WindowInsets.Type.statusBars())
        } else {
            window.setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN
            )
        }
        supportActionBar?.hide()
    }

    private fun observeSession() {
        viewModel.getSession().observe(this) { user ->
            if (user != null) {
                showLoading(false)
                displayLoggedIn(user)
            } else {
                showLoading(true)
                navigateToGetStarted()
            }

        }
    }

    private fun navigateToGetStarted() {
        val intent = Intent(this, GetStartedActivity::class.java)
        startActivity(intent)
        finish()
    }

    private fun displayLoggedIn(user: UserModel) {
        binding.cardViewProfile.visibility = View.VISIBLE
        binding.cardViewWorkout.visibility = View.VISIBLE
        binding.tvUsername.text = user.token
        displayCurrentDate()
    }
    private fun displayCurrentDate() {
        val sdf = SimpleDateFormat("dd MMM yyyy", Locale.US)
        val currentDate = Date()
        binding.tvCurrentDate.text = sdf.format(currentDate)
    }

    private fun showLoading(isLoading : Boolean) {
        binding.progressBar.visibility = if (isLoading) View.VISIBLE else View.INVISIBLE
    }

}